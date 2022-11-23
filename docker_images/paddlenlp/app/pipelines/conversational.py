from typing import Any, Dict, List, Union

from app.pipelines import Pipeline
from paddlenlp.taskflow import Taskflow


class ConversationalPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.pipeline = Taskflow("dialogue", task_path=model_id, from_hf_hub=True)

    def __call__(self, inputs: Dict[str, Union[str, List[str]]]) -> Dict[str, Any]:
        """
        Args:
            inputs (:obj:`dict`): a dictionary containing the following key values:
                text (`str`, *optional*):
                    The initial user input to start the conversation
                past_user_inputs (`List[str]`, *optional*):
                    Eventual past history of the conversation of the user. You don't need to pass it manually if you use the
                    pipeline interactively but if you want to recreate history you need to set both `past_user_inputs` and
                    `generated_responses` with equal length lists of strings
                generated_responses (`List[str]`, *optional*):
                    Eventual past history of the conversation of the model. You don't need to pass it manually if you use the
                    pipeline interactively but if you want to recreate history you need to set both `past_user_inputs` and
                    `generated_responses` with equal length lists of strings
        Return:
            A :obj:`dict`: a dictionary containing the following key values:
                generated_text (`str`):
                    The answer of the bot
                conversation   (`Dict[str, List[str]]`):
                    A facility dictionnary to send back for the next input (with the new user input addition).

                        past_user_inputs (`List[str]`)
                            List of strings. The last inputs from the user in the conversation, after the model has run.
                        generated_responses	(`List[str]`)
                            List of strings. The last outputs from the model in the conversation, after the model has run.
        """
        text = inputs["text"]
        past_user_inputs = inputs.get("past_user_inputs", [])
        generated_responses = inputs.get("generated_responses", [])
        complete_message_history = []
        for user_input, responses in zip(past_user_inputs, generated_responses):
            complete_message_history.extend([user_input, responses])
        complete_message_history.append(text)
        cur_response = self.pipeline(complete_message_history)[0]
        past_user_inputs.append(text)
        generated_responses.append(cur_response)
        return {
            "generated_text": cur_response,
            "conversation": {
                "generated_responses": generated_responses,
                "past_user_inputs": past_user_inputs,
            },
        }
