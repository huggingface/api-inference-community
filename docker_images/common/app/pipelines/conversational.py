from typing import Any, Dict, List, Union

from app.pipelines import Pipeline


class ConversationalPipeline(Pipeline):
    def __init__(self, model_id: str):
        # IMPLEMENT_THIS
        # Preload all the elements you are going to need at inference.
        # For instance your model, processors, tokenizer that might be needed.
        # This function is only called once, so do all the heavy processing I/O here
        raise NotImplementedError(
            "Please implement ConversationalPipeline __init__ function"
        )

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
                    A facility dictionary to send back for the next input (with the new user input addition).

                        past_user_inputs (`List[str]`)
                            List of strings. The last inputs from the user in the conversation, after the model has run.
                        generated_responses	(`List[str]`)
                            List of strings. The last outputs from the model in the conversation, after the model has run.
        """
        # IMPLEMENT_THIS
        raise NotImplementedError(
            "Please implement ConversationalPipeline __call__ function"
        )
