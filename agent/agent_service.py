import requests
import datetime
import json
import time
from flask import request

from agent import QwAgent


class RestApi():
    def __init__(self, agent, llm_url="http://192.168.186.18:3335/llm_test/predict"):
        self.agent = agent
        self.critic_agent = QwAgent()
        self.critic_prompt = """user:{user_input}\n\nagent:{agent_response}\n\nNow you're the agent's supervisor and please judge whether the answer given by agent to user is reasonable or correct. Use the following format:Result: the resutl to chose, should be one of ['right', 'error']\nReason: the result reason"""
        self.agent_map = {"core_agent": self.agent, "critic_agent": self.critic_agent}
        self.llm_url = llm_url
        self.TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""
        self.REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

        {tool_descs}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {query}"""

    @staticmethod
    def api_check():
        """
        get请求，用于验证服务是否正常运行
        :return:
        """
        return "OK"

    def api_chat(self):
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
                inputs = data['input']  # 获取用户输入
                is_critic = data.get('is_critic', False)
                print("user input->\n\n", inputs)
                prompt = self.create_prompt(inputs)
                print("model input->\n\n", prompt)
                rst = self.chat(prompt)
                print("model output->\n\n", rst)
                rst = self.tool_parse(rst)
                print("agent output->\n\n", rst)
                if is_critic:
                    # critic agent评判
                    critic_res = self.critic(inputs, rst)
                    print("critic_res->\n\n", critic_res)
                    critic_result, critic_reason = self.critic_parse(critic_res)
                    print("critic_result—>", critic_result)
                    print("critic_reason—>", critic_reason)

                    ret = {"code": 1, "agent_result": rst, "critic_result": critic_result, "critic_reason": critic_reason}
                else:
                    ret = {"code": 1, "agent_result": rst}
                return json.dumps(ret, ensure_ascii=False)
            else:
                return "No doc paramater in json"
        else:
            return "We only allow POST method. "

    def create_prompt(self, query):
        tool_descs = []
        tool_names = []
        for info in self.agent.tools:
            tool_descs.append(
                self.TOOL_DESC.format(
                    name_for_model=info['name_for_model'],
                    name_for_human=info['name_for_human'],
                    description_for_model=info['description_for_model'],
                    parameters=json.dumps(
                        info['parameters'], ensure_ascii=False),
                )
            )
            tool_names.append(info['name_for_model'])
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool_names)

        prompt = self.REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names, query=query)
        print("prompt", prompt)
        return prompt

    def critic(self, user_input, agent_response):
        prompt = self.critic_prompt.format_map({"user_input": user_input, "agent_response": agent_response})
        print("critic input->\n\n", prompt)
        return self.chat(prompt)

    def chat(self, prompt):
        data = {"input": prompt}
        response = requests.post(self.llm_url, data=json.dumps(data))
        if response.status_code == 200:
            response = json.loads(response.text)['result']
            print("response->\n\n", response)
            return response
        return {"post error": response.status_code}

    def tool_parse(self, response):
        """
        解析是否需要调用某个函数，由代理来实现调用，并返回调用结果
        Args:
            response: llm response

        Returns:
            函数调用结果
        """
        if "Observation:" in response:
            # 解析
            try:
                parse_response = [_ for _ in response.split('Observation:')[0].split('\n') if "Action" in _]
                action_name = parse_response[0].split(':')[1].strip()
                action_input = eval(parse_response[1].split('Action Input:')[-1])
                print("解析得到的结果", action_name, action_input)
            except Exception as e:
                return {"parse error": e}
            if action_name in self.agent.tool_map:
                print("action_input->\n\n", action_input)
                action_result = self.agent.tool_map.get(action_name)(**action_input)
            else:
                action_result = response.split('Final Answer:')[-1].strip()
        else:
            action_result = response.split('Final Answer:')[-1].strip()
        return action_result

    def critic_parse(self, response):
        if "Result" and "Reason" in response:
            response = response.split('\n')
            result = response[0].split(': ')[1]
            reason = response[1].split(': ')[1]
            return result, reason
        else:
            return "unknow", {"error critic": "无法评判"}

    def create_agent(self, agent_role, agent_template):
        if agent_role not in self.agent_map:
            self.agent_map[agent_role] = QwAgent()
            self.agent_map["{}-template".format(agent_role)] = agent_template


def run_deploy(args):
    """
    发布阶段
    :param args: 步骤对应全部参数
    :return:
    """
    import shutil
    from flask_cors import CORS
    from flask import Flask
    from agent import QwAgent

    # set restful server
    app = Flask(__name__)
    CORS(app)  # 跨域设置

    agent = QwAgent()
    rest_api = RestApi(agent=agent)

    app.add_url_rule("/" + args["serviceName"] + "/check", None, getattr(rest_api, "api_check"), methods=['GET'])
    app.add_url_rule("/" + args["serviceName"] + "/chat", None, getattr(rest_api, "api_chat"),
                     methods=['POST'])

    # 启动服务
    app.run(host="0.0.0.0", port=int(args["port"]), debug=False)
    return "OK"


if __name__ == "__main__":
    args = {"port": 3334,
            "serviceName": "intelligent_agent"}
    run_deploy(args)
