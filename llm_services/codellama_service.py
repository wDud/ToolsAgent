from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import datetime
import json
import time
import torch
from flask import request


class RestApi():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.chat = "{inputs}"

    def inference(self, prompt):
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(inputs, max_length=256, top_k=1)
        response = self.tokenizer.decode(outputs[0])
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

    model_path = args['model_path']
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device='cuda')
    model = model.eval()

    rest_api = RestApi(model=model, tokenizer=tokenizer)

    app.add_url_rule("/" + args["serviceName"] + "/check", None, getattr(rest_api, "api_check"), methods=['GET'])
    app.add_url_rule("/" + args["serviceName"] + "/predict", None, getattr(rest_api, "api_predict"),
                     methods=['POST'])

    # 启动服务
    app.run(host="0.0.0.0", port=int(args["port"]), debug=False, threaded=False)
    return "OK"


if __name__ == "__main__":
    args = {"port": 3336,
            "serviceName": "codellama_test",
            "model_path": "/root/llama/CodeLlama-13b-Python-hf"}
    run_deploy(args)
