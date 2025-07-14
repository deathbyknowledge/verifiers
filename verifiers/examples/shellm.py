import json
from abc import abstractmethod
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Union

from openai import AsyncOpenAI, OpenAI
import verifiers as vf
from verifiers import ChatMessage, Messages, MultiTurnEnv

from dotenv import load_dotenv
load_dotenv()


import docker
import time
import socket

class Sandbox:
    """Manages an isolated Docker container with a persistent shell session."""
    
    def __init__(self, image="shellm-sandbox:latest", setup_commands=[]):
        self.image = image
        self.client = docker.from_env()
        self.container = None
        self.socket = None
        self.command_id = 0
        self.setup_commands = " && ".join(setup_commands).replace("'", "'\\''")

    def start(self):
        """Starts a new Docker container and sets up a persistent shell session."""
        print("Starting secure sandbox...")
        try:
            # Start container with bash as the main process
            self.container = self.client.containers.run(
                self.image,
                command="/bin/bash",
                tty=True,
                stdin_open=True,
                detach=True
            )
            # Install tools using exec_run
            print("Installing tools in sandbox...")
            exit_code, (stdout, stderr) = self.container.exec_run(
                f"/bin/bash -c '{self.setup_commands}'", demux=True
            )
            if exit_code != 0:
                raise Exception(f"Sandbox setup failed: {stderr.decode() if stderr else 'Unknown error'}")
            print("Sandbox ready.")
            # Attach to the bash process
            self.socket = self.container.attach_socket(
                params={'stdin': 1, 'stdout': 1, 'stderr': 1, 'stream': 1}
            )
            self.socket._sock.settimeout(1)  # Set timeout for socket reads
            self.socket._sock.send(b'stty -echo\n')
            time.sleep(0.1)

        except Exception as e:
            print(f"Error starting sandbox: {e}")
            self.stop()
            raise

    def execute_command(self, command: str):
        """Executes a command in the persistent shell session."""
        if not self.container or not self.socket:
            raise Exception("Sandbox is not running or session is not started.")

        self.command_id += 1
        id = self.command_id
        stdout_file = f"/tmp/stdout_{id}.txt"
        stderr_file = f"/tmp/stderr_{id}.txt"
        exitcode_file = f"/tmp/exitcode_{id}.txt"
        marker = f"COMMAND_DONE_{id}"

        # By wrapping the command in `{ ...; }`, we create a command group.
        # This allows shell redirections inside the `command` (e.g., `> file.txt`)
        # to function correctly, while we capture the output of the group itself.
        # Unlike a subshell `(...)`, this executes in the current shell context,
        # so commands like `cd` work as expected.
        # The spaces and trailing semicolon are required syntax for the shell group.
        grouped_command = f"{{ {command}; }}"

        # Send command with output redirection and marker
        cmd_to_send = (
            f"{grouped_command} > {stdout_file} 2> {stderr_file}; "
            f"echo $? > {exitcode_file}; echo '{marker}'\n"
        )
        self.socket._sock.send(cmd_to_send.encode('utf-8'))

        # Wait for command completion
        self.read_until_marker(marker)

        # Read output files
        stdout_exit, (stdout_data, stdout_errdata) = self.container.exec_run(f"cat {stdout_file}", demux=True)
        stderr_exit, (stderr_data, stderr_errdata) = self.container.exec_run(f"cat {stderr_file}", demux=True)
        exitcode_exit, (exitcode_data, exitcode_errdata) = self.container.exec_run(f"cat {exitcode_file}", demux=True)
        
        if stdout_exit != 0:
            raise Exception(f"Command failed: {stdout_errdata.decode('utf-8')}")
        if stderr_exit != 0:
            raise Exception(f"Command failed: {stderr_errdata.decode('utf-8')}")
        if exitcode_exit != 0:
            raise Exception(f"Command failed: {exitcode_errdata.decode('utf-8')}")

        # Decode outputs
        stdout = stdout_data.decode('utf-8') if stdout_data else ""
        stderr = stderr_data.decode('utf-8') if stderr_data else ""
        exit_code_str = exitcode_data.decode('utf-8').strip() if exitcode_data else "0"

        # Parse exit code
        try:
            exit_code = int(exit_code_str)
        except ValueError:
            exit_code = -1  # Indicate parsing error

        # Clean up temporary files
        self.container.exec_run(f"rm {stdout_file} {stderr_file} {exitcode_file}")

        return stdout, stderr, exit_code

    def read_until_marker(self, marker, timeout=10):
        """Reads from the socket until the specified marker is found."""
        if self.socket is None:
            raise Exception("Socket not initialized")
        accumulated = ""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                data = self.socket._sock.recv(1024).decode('utf-8')
                accumulated += data
                if marker in accumulated:
                    return accumulated
            except socket.timeout:
                continue
        raise TimeoutError(f"Timeout waiting for marker: {marker}")

    def stop(self):
        """Stops the shell session and removes the container."""
        if self.socket:
            try:
                self.socket._sock.send(b"exit\n")
                time.sleep(1)  # Allow bash to exit
            except Exception as e:
                print(f"Error sending exit command: {e}")
            self.socket.close()
            self.socket = None
        if self.container:
            try:
                self.container.remove(force=True)
            except docker.errors.APIError as e: # type: ignore
                print(f"Warning: Could not stop container properly: {e}")
            self.container = None



import random
from typing import Tuple, List, Dict, Any

from datasets import Dataset, load_dataset

from verifiers import MultiTurnEnv, Parser, Rubric
import os


class ShellEnv(MultiTurnEnv):
    """
    """
    def __init__(self,
                 num_samples: int = 1000,
                 num_eval_samples: int = 100,
                 seed: int = 0,
                 **kwargs):
        self.seed = seed
        dataset = load_dataset('deathbyknowledge/V3-shell-format', split=['train'])
        eval_dataset = Dataset.from_list(dataset[200:300]) # type: ignore
        dataset = Dataset.from_list(dataset[300:]) # type: ignore

        parser = Parser()
        base_url = "https://api.deepseek.com"
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        rubric = ShellJudgeRubric(parser=parser, judge_model="deepseek-chat", judge_client=OpenAI(base_url=base_url, api_key=api_key))

        def check_exit_codes(completion, answer, state, info, **kwargs) -> float:
            exit_codes = state['exit_codes']
            # Every non-zero exit code is penalized with -0.05 reward
            return -1 * sum([0.05 for x in exit_codes if x != 0])
            return is_correct / (num_turns + 1)
        def check_success_command(completion, answer, state, info, **kwargs) -> float:
            success_command = info['success_condition']
            sandbox = state['sandbox']
            _, _, exit_code = sandbox.execute_command(success_command)
            sandbox.stop()
            # Success command has low reward as it's LLM generated and could be wrong,
            # so I reward it a bit for being correct but let the Judge LLM provide
            # stronger rewards.
            return 0.2 if exit_code == 0 else 0.0
        rubric.add_reward_func(check_exit_codes)
        rubric.add_reward_func(check_success_command)

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            parser=parser,
            rubric=rubric,
            message_type='chat',
            max_turns=30,
            **kwargs
        )
        self.parser = parser
        self.rubric = rubric

    def is_completed(self,
                     messages: Messages,
                     state: Dict[str, Any],
                     **kwargs: Any) -> bool:
        if isinstance(messages, str):
            raise ValueError("Messages must be a list of ChatMessages")
        if messages[-1]["role"] == "assistant" and messages[-1]["content"] == "exit 0":
            return True
        return False

    def env_response(self,
                     messages: Messages,
                     state: Dict[str, Any],
                     **kwargs: Any) -> Tuple[ChatMessage, Dict[str, Any]]:
        if isinstance(messages, str):
            raise ValueError("Messages must be a list of ChatMessages")
        # load active sandbox container
        sandbox: Sandbox = state['sandbox']
        # read command
        command = messages[-1]["content"]
        stdout, stderr, exit_code = sandbox.execute_command(command)
        state['exit_codes'].append(exit_code)
        feedback = ""
        if stdout:
            feedback += stdout
        if stderr:
            feedback += stderr
        env_message: ChatMessage = {"role": "user", "content": feedback}
        return env_message, state

    def format_dataset(self,
                       dataset: Dataset,
                       system_prompt: str | None = None,  # Ignored here since per-sample
                       few_shot: list[dict] | None = None,
                       question_key: str = "question",
                       answer_key: str = "answer") -> Dataset:
        """
        Override to use per-sample system prompts from "custom_system" column.
        Falls back to self.system_prompt if not present.
        """
        def format_prompt_fn(example: dict) -> dict:
            # Get per-sample system prompt, fallback to class-level
            parsed = json.loads(example.get("prompt", ""))
            task = parsed[0].get("content")
            info = {}
            setup_commands: List[str] = json.loads(example.get("setup_commands", "[]"))
            success_condition: str | None = json.loads(example.get("success_condition", "null"))
            info['setup_commands'] = setup_commands
            info['success_condition'] = success_condition
            
            messages = []
            if task:
                messages.append({"role": "system", "content": task})
            
            return {"prompt": messages, "info": info}
        
        return dataset.map(format_prompt_fn, num_proc=8)
        
    async def rollout(self,
                      client: AsyncOpenAI,
                      model: str,
                      prompt: Messages,
                      answer: str,
                      task: str = "default",
                      info: Dict[str, Any] = {},
                      sampling_args: Dict[str, Any] = {},
                      **kwargs: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate a multi-turn rollout with the environment (messages, state).
        """
        assert isinstance(prompt, list)
        messages = deepcopy(prompt) 
        is_completed = False
        setup_commands = info.get('setup_commands', [])
        sandbox = Sandbox(setup_commands=setup_commands)
        sandbox.start()
        state = {'answer': answer, 'responses': [], 'sandbox': sandbox, 'exit_codes': []}
        completion = []
        turn = 0
        while not is_completed:
            if self.is_completed(messages, state, **kwargs):
                is_completed = True
                break
            response = await self.get_model_response(
                prompt=messages,
                client=client,
                model=model,
                sampling_args=sampling_args,
                message_type="chat"
            )
            if not response:
                raise Exception("No response from model")
            response_text: str = response.choices[0].message.content # type: ignore
            messages.append({"role": "assistant", "content": response_text})
            completion.append({"role": "assistant", "content": response_text})
            state['responses'].append(response)
            turn += 1
            if self.is_completed(messages, state, **kwargs) or turn >= self.max_turns:
                is_completed = True
            else:
                env_msg, state = self.env_response(messages, state, **kwargs)
                messages.append(env_msg)
                completion.append(env_msg)
        return completion, state


DEFAULT_JUDGE_PROMPT = """
You are an expert evaluator. Your role is to determine if a task, executed in a Linux shell environment, was successfully completed based on the provided trajectory.
The agent MUST always provide reasoning for their actions by submitting bash comments. This is mandatory and does not discount the agent's score.
The agent must also use `exit 0` once the task is completed.
You will be given the initial task, any setup commands that were run, and the full sequence of shell interactions, and outputs from the AI agent.

Rate the trajectory on a scale from 1 to 5, where:
1 = Completely incorrect. The agent made no progress or took actions completely unrelated to the task.
2 = Mostly incorrect. The agent took some initial steps that were related to the task but failed to make substantial progress or went down a wrong path.
3 = Partially correct. The agent completed a significant portion of the task but had major errors or failed to complete all requirements.
4 = Mostly correct. The agent successfully completed the task but took unnecessary steps, had minor, recoverable errors, or the solution was inefficient.
5 = Perfectly correct. The agent completed the task efficiently and accurately, following best practices.

Provide a concise reasoning for your decision and then the final numeric rating.
Respond with a single JSON object containing 'reasoning' and 'rating'  keys.
"""

class ShellJudgeRubric(Rubric):
    def __init__(self,
                 judge_client: OpenAI | None = None,
                 judge_model: str = "gpt-4.1-nano",
                 judge_prompt: str = DEFAULT_JUDGE_PROMPT,
                 parser: Parser = Parser(),
                 **kwargs):
        super().__init__(**kwargs)
        self.judge_client = judge_client if judge_client is not None else OpenAI()
        self.judge_model = judge_model
        self.judge_prompt = judge_prompt
        self.parser = parser
        self.add_reward_func(self.judge_reward_func)


    def _format_history(self, completion: List[Dict[str, Any]], exit_codes: List[int]):
        """Formats the trajectory into a string for the prompt."""
        if not completion:
            return "No actions were taken."

        formatted = []
        for (turn, msg) in enumerate(completion):
            if msg['role'] == 'user':
                continue
            formatted.append(f"Turn {turn}:")
            formatted.append(f"Action:`{msg['content']}`")
            formatted.append(f"Exit Code: {exit_codes[turn]}")
            if turn < len(exit_codes) - 1:
                # assume next message is the observation
                formatted.append(f"Shell Output:\n---\n{completion[turn + 1]['content'].strip()}\n---")
        return "\n".join(formatted)

    def judge_reward_func(self, prompt, completion, answer, state, **kwargs) -> float:
        task = prompt[0]['content']
        setup_commands = state['setup_commands']
        exit_codes = state['exit_codes']

        history = self._format_history(completion, exit_codes)
        setup_str = "\n".join(f"$ {cmd}" for cmd in setup_commands) if setup_commands else "None"

        prompt = f"TASK: {task}\n\nSETUP COMMANDS:\n{setup_str}\n\nTRAJECTORY:\n{history}\n\nBased on the trajectory, was the task successfully completed? Provide your rating and reasoning."
        system_prompt = self.judge_prompt
        judge_response = self.judge_client.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
        )
        judge_response = json.loads(str(judge_response.choices[0].message.content))
        reasoning = judge_response['reasoning']
        rating = judge_response['rating']
        reward = 0
        # TODO: don't use yolo numbers
        if rating == 5:
            reward = 1.0
        elif rating == 4:
            reward = 0.4
        elif rating == 3:
            reward = 0.0
        elif rating == 2:
            reward = -0.2
        elif rating == 1:
            reward = -0.5
        return reward
    


    

"""
inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model deathbyknowledge/Qwen3-8B-Shell-SFT --tensor-parallel-size 1

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --config-file configs/zero3.yaml --num-processes 1 verifiers/examples/shellm.py
"""

model_name = f'deathbyknowledge/Qwen3-8B-Shell-SFT'
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = ShellEnv(
    num_samples=2000, 
    num_eval_samples=20
)

run_name = f"shell-grpo-8B"
training_args=vf.grpo_defaults(run_name=run_name)
training_args.num_iterations=1
training_args.per_device_train_batch_size=2
training_args.num_generations=8
training_args.gradient_accumulation_steps=6
training_args.max_prompt_length=1024
training_args.max_completion_length=3072
training_args.max_steps=100
training_args.mask_env_responses=True

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()
