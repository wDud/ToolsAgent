class Tools():
    def __init__(self):
        self.tools = []

    def add_tool(self, tool):
        self.tools.append(tool)
        print(f"已成功添加工具：{tool['name_for_human']}")


if __name__ == "__main__":
    tools = [
        {
            'name_for_human': '夸克搜索',
            'name_for_model': 'quark_search',
            'description_for_model': '夸克搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。',
            'parameters': [{
                'name': 'search_query',
                'description': '搜索关键词或短语',
                'required': True,
                'schema': {
                    'type': 'string'
                }}]},
        {
            'name_for_human': '通义万相',
            'name_for_model': 'image_gen',
            'description_for_model': '通义万相是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL',
            'parameters': [{
                'name': 'query',
                'description': '中文关键词，描述了希望图像具有什么内容',
                'required': True,
                'schema': {
                    'type': 'string'
                }}],
        }
    ]
    Tool = Tools()
    for tool in tools:
        Tool.add_tool(tool)
