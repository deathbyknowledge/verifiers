import os, sys
import random
from typing import Tuple, List, Dict, Any

from datasets import Dataset
import nltk 

# monkey-patch nltk.download to always be quiet before importing textarena
_original_nltk_download = nltk.download
nltk.download = lambda *args, **kwargs: _original_nltk_download(*args, **{**kwargs, 'quiet': True})

import textarena as ta 

from verifiers import (
    ChatMessage,
    State,
    MultiTurnEnv,
    XMLParser,
    Rubric
)


GUESS_SYSTEM_PROMPT = """You are a competitive game player. \
Make sure you read the game instructions carefully, and always follow the required format.

In each turn, think step-by-step inside <think>...</think> tags, \
then follow the instructions inside <guess>...</guess> tags."""

class TextArenaEnv(MultiTurnEnv):
    """
    Wrapper for TextArena environments.

    Supported games:
    - Wordle-v0
    - Hangman-v0
    """
    def __init__(self,
                 game: str = "Wordle-v0",
                 num_samples: int = 1000,
                 num_eval_samples: int = 100,
                 seed: int = 0,
                 **kwargs):
        self.game = game
        self.num_samples = num_samples
        self.num_eval_samples = num_eval_samples
        self.seed = seed

        nltk.download('words', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        dataset, eval_dataset = self.ta_to_hf()
        parser = XMLParser(fields=["think", "guess"], answer_field="guess")
        rubric = Rubric(parser=parser)
        def check_answer_reward_func(completion, answer, **kwargs) -> float:
            guess = self.parser.parse_answer(completion)
            return 1.0 if guess == '[' + answer + ']' else 0.0
        def count_turns_reward_func(completion, answer, **kwargs) -> float:
            num_turns = len([x for x in completion if x['role'] == 'assistant'])
            is_correct = check_answer_reward_func(completion, answer, **kwargs)
            return is_correct / (num_turns + 1)
        rubric.add_reward_func(check_answer_reward_func)
        rubric.add_reward_func(count_turns_reward_func)
        rubric.add_reward_func(parser.get_format_reward_func(), weight=0.2)

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=GUESS_SYSTEM_PROMPT,
            parser=parser,
            rubric=rubric,
            message_type='chat',
            **kwargs
        )
        self.parser = parser
        self.rubric = rubric

    def is_completed(self,
                     messages: List[ChatMessage],
                     state: State,
                     **kwargs: Any) -> bool:
        if 'is_finished' in state and state['is_finished']:
            state.pop('ta_env')
            return state['is_finished']
        return False

    def env_response(self,
                     messages: List[ChatMessage],
                     state: State,
                     **kwargs: Any) -> Tuple[ChatMessage, State]:
        # load env 
        if 'ta_env' not in state:
            ta_env = ta.make(env_id=self.game)
            ta_env.reset(num_players=1)
            ta_env.state.game_state['secret_word'] = state['answer']
            state['ta_env'] = ta_env
        else:
            ta_env = state['ta_env']
        # parse guess
        turn = self.parser.parse(messages[-1]["content"])
        guess = turn.guess
        # step env
        is_finished, _ = ta_env.step(str(guess))
        state['is_finished'] = is_finished
        _, observation = ta_env.get_observation()
        if "Feedback:" in observation:
            feedback = observation.split("Feedback:")[-1]
        else:
            feedback = observation
        env_message: ChatMessage = {"role": "user", "content": str(feedback)}
        return env_message, state

    def ta_to_hf(self) -> Tuple[Dataset, Dataset]:
        dataset_rows = []
        eval_dataset_rows = []
        ta_env = ta.make(env_id=self.game)
        ta_env.reset(num_players=1)
        _, user_prompt = ta_env.get_observation()
        words = ta_env.word_list
        # set seed 
        random.seed(self.seed)
        for i in range(self.num_samples + self.num_eval_samples):
            question = user_prompt
            answer = random.choice(words)
            if i < self.num_samples:
                dataset_rows.append({
                    "question": question,
                    "answer": answer
                })
            else:
                eval_dataset_rows.append({
                    "question": question,
                    "answer": answer
                })
        dataset = Dataset.from_list(dataset_rows)
        eval_dataset = Dataset.from_list(eval_dataset_rows)
        return dataset, eval_dataset