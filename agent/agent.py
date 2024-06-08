"""
qing wen agent
使用通义千问作为 Intelligent agent

1、user interaction api is qw model api
2、qw model decide whether use tools
3、parse function
"""
import json

from plug_tool import Tools
import requests


class QwAgent(Tools):
    def __init__(self):
        super().__init__()
        init_tool = {
            'name_for_human': '插件启用',
            'name_for_model': 'plug_add',
            'description_for_model': '插件启动是一个功能扩展服务，输入要开启的插件名称，开启该插件功能用于用户后续使用',
            'parameters': [{
                'name': 'fuction_name',
                'description': '函数名称，描述了希望添加哪个函数到可用工具中',
                'required': True,
                'schema': {
                    'type': 'string'
                }}]
        }
        self.add_tool(init_tool)
        self.add_compute_des = {
            'name_for_human': '加法计算器',
            'name_for_model': 'add_compute',
            'description_for_model': '加法计算器是一个算数计算服务，输入两个数，返回这两个数之和',
            'parameters': [{
                'name': 'a',
                'description': '加数',
                'required': True,
                'schema': {
                    'type': 'float'
                }}, {'name': 'b', 'description': '被加数', 'required': True, 'schema': {
                'type': 'float'
            }}]
        }
        self.multiply_compute_des = {
            'name_for_human': '乘法计算器',
            'name_for_model': 'multiply_compute',
            'description_for_model': '乘法计算器是一个算数计算服务，输入两个数，返回这两个数的乘积',
            'parameters': [{
                'name': 'a',
                'description': '乘数',
                'required': True,
                'schema': {
                    'type': 'float'
                }}, {'name': 'b', 'description': '被乘数', 'required': True, 'schema': {
                'type': 'float'
            }}]
        }
        self.web_search_des = {
            'name_for_human': '互联网搜索',
            'name_for_model': 'web_search',
            'description_for_model': '互联网搜索是一个上网搜索服务，输入搜索关键词，返回关键词相关的网页信息',
            'parameters': [{
                'name': 'key',
                'description': '中文关键词，希望知道相关的信息',
                'required': True,
                'schema': {
                    'type': 'str'
                }}]
        }
        self.coding_des = {
            'name_for_human': '代码编写',
            'name_for_model': 'coding',
            'description_for_model': '代码编写是一个代码编写服务，输入功能需求以及代码实现语言，返回要求语言实现功能的代码',
            'parameters': [{
                'name': 'language',
                'description': '英文 编程语言的名字',
                'required': True,
                'schema': {
                    'type': 'str'
                }},{
                'name': 'intent',
                'description': '英文 需要实现的功能翻译为英文作为参数',
                'required': True,
                'schema': {
                    'type': 'str'
                }}]
        }
        self.tool_map = {'add_compute': self._parser_add_compute,
                         "add_compute-des": self.add_compute_des,
                         'multiply_compute': self._parser_multiply_compute,
                         "multiply_compute-des": self.multiply_compute_des,
                         'web_search': self._parser_web_search,
                         "web_search-des": self.web_search_des,
                         "coding": self._parser_coding,
                         "coding-des": self.coding_des,
                         "plug_add": self._plug_add}

    def _parser_add_compute(self, **kwargs):
        """
        简单两项数加法工具
        Args:
            a: 加数
            b: 被加数
        Returns:
            两项数之和
        """
        a = kwargs.get('a')
        b = kwargs.get('b')
        return a + b

    def _parser_multiply_compute(self, **kwargs):
        """
        简单两项数乘法工具
        Args:
            a: 乘数
            b: 被乘数
        Returns:
            两项数之积
        """
        a = kwargs.get('a')
        b = kwargs.get('b')
        return a * b

    def _parser_web_search(self, **kwargs):
        search_query = kwargs.get('key')
        url = f"https://baike.baidu.com/search/none?word={search_query}"
        response = requests.get(url)
        if response.status_code == 200:
            search_result = response.content.decode('utf-8')
            return search_result
        return {"post error", response.status_code}

    def _parser_coding(self, **kwargs):
        language = kwargs.get('language')
        intent = kwargs.get('intent')
        data = {
            "instruction": "",
            "input": f"# language: {language}\n# {intent}",
            "mode": "chat"
        }
        res = requests.post("http://192.168.186.18:3336/codegeex_test/predict",
                            data=json.dumps(data, ensure_ascii=False))
        if res.status_code == 200:
            return json.loads(res.text)
        else:
            return {"coding post error": res.status_code}

    def _plug_add(self, **kwargs):
        function_name = kwargs.get("function_name")
        print("要添加的插件函数为->", function_name)
        if function_name in self.tool_map:
            tool_des = self.tool_map["{}-des".format(function_name)]
            if tool_des in self.tools:
                return "当前插件已开启，请勿重复开启"
            self.add_tool(tool_des)
            return "添加成功"
        else:
            return "抱歉，当前插件功能暂未实现，请稍后再试！"


if __name__ == "__main__":
    agent = QwAgent()
