import argparse

def parse_opt():
    '''解析命令行参数'''
    parser = argparse.ArgumentParser()
    # 定义参数
    parser.add_argument('--name', type=str, default='neo',
                        help='Choose LLM')
    parser.add_argument('--dataset', type=str, default='vqacpv2',
                        help='Choose dataset')

    args = parser.parse_args()

    return args