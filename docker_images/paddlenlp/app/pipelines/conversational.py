from http.client import responses
from typing import Any, Dict, List, Union

from app.pipelines import Pipeline
from paddlenlp.taskflow import Taskflow


class ConversationalPipeline(Pipeline):
    def __init__(self, model_id: str):
        # TODO: how to setup params
        self.pipeline = Taskflow("dialogue", task_path=model_id, from_hf_hub=True)
        

    def __call__(self, inputs: Dict[str, Union[str, List[str]]]) -> Dict[str, Any]:
        """
        Args:
            inputs (:obj:`dict`): a dictionnary containing the following key values:
                text (`str`, *optional*):
                    The initial user input to start the conversation. If not provided, a user input needs to be provided
                    manually using the [`~Conversation.add_user_input`] method before the conversation can begin.
                past_user_inputs (`List[str]`, *optional*):
                    Eventual past history of the conversation of the user. You don't need to pass it manually if you use the
                    pipeline interactively but if you want to recreate history you need to set both `past_user_inputs` and
                    `generated_responses` with equal length lists of strings
                generated_responses (`List[str]`, *optional*):
                    Eventual past history of the conversation of the model. You don't need to pass it manually if you use the
                    pipeline interactively but if you want to recreate history you need to set both `past_user_inputs` and
                    `generated_responses` with equal length lists of strings
        Return:
            A :obj:`dict`:. ???
        """
        text = inputs["text"]
        past_user_inputs = inputs["past_user_inputs"]
        generated_responses = inputs["generated_responses"]
        complete_message_history = []
        for user_input, responses in zip(past_user_inputs, generated_responses):
            complete_message_history.extend([user_input, responses])
        complete_message_history.append(text)
        response = self.pipeline(complete_message_history)

        ## To be implemented once the API clears up



        