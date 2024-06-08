from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

import datetime
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LlamaForCausalLM
from peft import PeftModel, PeftConfig
import torch
from flask import request


class RestApi():
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.chat = "{inputs}"
        self.history = None

    def inference(self, question):
        response, history = self.model.chat(self.tokenizer, question, history=self.history)
        print("self.history", self.history)
        self.history = history
        return response

    @staticmethod
    def api_check():
        """
        get请求，用于验证服务是否正常运行
        :return:
        """
        return "OK"

    def api_predict(self):
        if request.method == 'POST':
            try:
                ip = request.headers['X-Forwarded-For']
            except KeyError:
                ip = request.remote_addr
            print("==============访问predict开始=============")
            print("当前访问时间为:", datetime.datetime.now())
            print("当前访问ip地址为:", ip)
            stt = time.time()
            data = json.loads(request.data.decode('utf-8'))
            if "input" in data:
                inputs = data['input']
                history = data.get('history')
                if history != 'true':
                    self.history = None
                print("model input\n\n", inputs)
                rst = self.inference(inputs)
                print("model output\n\n", rst)
                ret = {"code": 1, "result": rst}
                return json.dumps(ret, ensure_ascii=False)
            else:
                return "No doc paramater in json"
        else:
            return "We only allow POST method. "


def run_deploy(args):
    """
    发布阶段
    :param args: 步骤对应全部参数
    :return:
    """
    import shutil
    from flask_cors import CORS
    from flask import Flask
    # set restful server
    app = Flask(__name__)
    CORS(app)  # 跨域设置
    if args['is_lora']:
        lora_model_path = args['lora_model_path']
        peft_config = PeftConfig.from_pretrained(lora_model_path)

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, trust_remote_code=True)

        # 结合基础模型和微调结果，加载模型
        model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, trust_remote_code=True,
                                                     load_in_4bit=True, torch_dtype=torch.float32, pretraining_tp=1)
        model = PeftModel.from_pretrained(model, lora_model_path)
    else:
        # 加载模型
        model_path = args['model_path']
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # use bf16
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, use_bf16=True).eval()
        # use fp16
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, use_fp16=True).eval()
        # use fp32
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",
                                                     trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(model_path,
                                                                   trust_remote_code=True)  # 可指定不同的生成长度、top_p等相关超参

    device = args['device']
    model.to(device)

    rest_api = RestApi(model=model, tokenizer=tokenizer, device=device)

    app.add_url_rule("/" + args["serviceName"] + "/check", None, getattr(rest_api, "api_check"), methods=['GET'])
    app.add_url_rule("/" + args["serviceName"] + "/predict", None, getattr(rest_api, "api_predict"),
                     methods=['POST'])

    # 启动服务
    app.run(host="0.0.0.0", port=int(args["port"]), debug=False, threaded=False)
    return "OK"


if __name__ == "__main__":
    args = {"port": 3335,
            "serviceName": "llm_test",
            "device": "cuda:0",
            "is_lora": False,
            "lora_model_path": "",
            "model_path": "/root/llama/Qwen-7B-Chat"}
    run_deploy(args)
