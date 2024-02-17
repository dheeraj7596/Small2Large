import io
import json

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def jsonl_load(f, mode="r"):
    """Load a .jsonl file into a dictionary."""
    f = _make_r_io_base(f, mode)
    json_list = []
    for line in f:
        json_list.append(json.loads(line))
    f.close()
    return json_list

def _concat_messages(messages):
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + message["content"].strip() + "\n"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    return message_text
