import numpy as np
import networkx as nx
from scapy.all import *
from scapy.layers.inet import TCP, IP, UDP
import random
import sys


class Net:

    def __init__(self):
        # self.name = name
        self.node_list = []  # 存储生成的所有节点
        self.path_list = []  # 存储生成的所有路径

    class Node(object):

        def __init__(self, name):
            self.name = name  # 表示节点名称
            self.up_nodes = []  # 表示该节点的所有上跳邻接节点集合
            self.down_nodes = []  # 表示该节点的所有下跳邻接节点集合
            self.down_weights = []  # 表示该节点的所有下跳邻接节点链路权重集合
            self.messages = []  # 表示该节点待处理的消息集合M
            self.edges = []  # 表示该节点下跳可以选择的链路集合

    class Path(object):

        def __init__(self, path_nodes):
            self.nodes = path_nodes

    class Edge(object):
        edge_list = []

        def __init__(self, upNode, downNode, weight):
            self.upNode = upNode  # 链路的起始节点
            self.downNode = downNode  # 链路的到达节点
            self.weight = weight  # 链路带宽,即链路能最多一次性传输的包数
            Net.Edge.edge_list.append(self)

    class Msg(object):
        def __init__(self, datapacket, now_node, prior, end_time=6):
            self.end_time = end_time  # 消息的有效截止时限ED
            self.r_endtime = end_time  # 消息模拟调度获取reward的有效截止时限
            self.deadline = end_time
            self.now_node = now_node  # 消息此时所处的节点
            self.next_node = None  # 消息即将发送去的节点
            self.initial_prior = prior  # 消息的优先级Pri
            self.prior = prior  # 消息的优先级Pri
            self.packet = datapacket  # 消息携带的网络数据包
            self.path = None  # 消息所要进行传输的路径

    def get_data(self, firstnode):
        dpkt = sniff(offline='testdata2.pcap')
        dpkt = list(dpkt)
        random.shuffle(dpkt)  # 打乱数据包的顺序
        endtimes = [2, 2, 3, 3]
        priors = [4, 3, 2, 1]
        for packet in dpkt:
            tos = packet[IP].tos / 4
            if tos == 8 or tos == 7:  # VOIP,优先级最高
                packet[IP].tos = 1
                endtime = endtimes[0]
                prior = priors[0]
            elif tos == 6 or tos == 5:  # 音视频流
                packet[IP].tos = 2
                endtime = endtimes[1]
                prior = priors[1]
            elif tos == 4 or tos == 3:  # 网页浏览,聊天,邮件
                packet[IP].tos = 3
                endtime = endtimes[2]
                prior = priors[2]
            else:  # P2P,FTP
                packet[IP].tos = 4
                endtime = endtimes[3]
                prior = priors[3]
            message = Net.Msg(packet, firstnode, prior, end_time=endtime)
            firstnode.messages.append(message)  # 每个包组装成一个消息实例,加入到0跳节点待处理的消息集合中
        return dpkt

    def read_graph(self, node_list, path_list):
        G = nx.read_gml("myTopo.txt")  # Poland, ANS，DFN
        nodes = list(G.nodes())  # 从拓扑图中获取到的所有节点名称
        edges = list(G.edges())  # 从拓扑图中获取到的所有链路，由源节点和目的节点构成
        for i in range(12):
            name = "n"+str(i)
            for node_name in nodes:
                if node_name == name:
                    node_name = Net.Node(node_name)
                    node_list.append(node_name)

        n0 = node_list[0]
        n1 = node_list[1]
        n2 = node_list[2]
        n3 = node_list[3]
        n4 = node_list[4]
        n5 = node_list[5]
        n6 = node_list[6]
        n7 = node_list[7]
        n8 = node_list[8]
        n9 = node_list[9]
        n10 = node_list[10]
        n11 = node_list[11]
        n0.down_nodes.append(n1)
        n0.down_weights.append(2)
        n0.down_nodes.append(n2)
        n0.down_weights.append(2)
        n0.down_nodes.append(n3)
        n0.down_weights.append(1)
        n1.down_nodes.append(n4)
        n1.down_weights.append(1)
        n1.down_nodes.append(n5)
        n1.down_weights.append(2)
        n2.down_nodes.append(n6)
        n2.down_weights.append(2)
        n2.down_nodes.append(n7)
        n2.down_weights.append(1)
        n3.down_nodes.append(n7)
        n3.down_weights.append(1)
        n4.down_nodes.append(n8)
        n4.down_weights.append(1)
        n5.down_nodes.append(n8)
        n5.down_weights.append(2)
        n5.down_nodes.append(n6)
        n5.down_weights.append(1)
        n6.down_nodes.append(n9)
        n6.down_weights.append(2)
        n7.down_nodes.append(n10)
        n7.down_weights.append(1)
        n8.down_nodes.append(n11)
        n8.down_weights.append(2)
        n9.down_nodes.append(n11)
        n9.down_weights.append(2)
        n10.down_nodes.append(n9)
        n10.down_weights.append(1)
        n10.down_nodes.append(n11)
        n10.down_weights.append(1)

        # for node_name in nodes:
        #     node_name = Net.Node(node_name)
        #     node_list.append(node_name)
        # for node in node_list:
        #     for edge in edges:
        #         if edge[0] == node.name:
        #             for next_node in node_list:
        #                 if edge[1] == next_node.name:
        #                     weight = G.get_edge_data(edge[0], edge[1])['weight']  # 获取链路权重
        #                     reverse = G.get_edge_data(edge[0], edge[1])['reverse']  # 获取链路正反方向
        #                     newedge = Net.Edge(node, next_node, weight)
        #                     if not reverse:
        #                         node.edges.append(newedge)
        #                         node.down_nodes.append(next_node)
        #                         node.down_weights.append(weight)
        #                         next_node.up_nodes.append(node)
        #                     else:
        #                         next_node.edges.append(newedge)
        #                         next_node.down_nodes.append(node)
        #                         next_node.down_weights.append(weight)
        #                         node.up_nodes.append(next_node)
        sourcenode = node_list[0]
        path_nodes = []
        depth = 0
        path_num, path_list = Net.getPath(self, sourcenode, path_nodes, node_list, path_list, depth)
        return node_list, path_list

    def getPath(self, startnode, path_nodes, node_list, path_list, depth):
        depth += 1
        # print(depth)
        if startnode.down_nodes:
            for downnode in startnode.down_nodes:  # 遍历该节点下所有子节点
                path_nodes.append(downnode)
                if downnode == node_list[-1]:  # 如果下跳节点是目的节点
                    newpathnodes = [x for x in path_nodes]
                    path = Net.Path(newpathnodes)  # 生成新路径
                    path_list.append(path)
                else:
                    Net.getPath(self, downnode, path_nodes, node_list, path_list, depth)
                path_nodes.remove(downnode)
        return len(path_list), path_list

    def schedule(self, action, count, s, node_list, path_list, remove_count):  # count为了平衡调度和输入的速度，等于路径的数量
        sourcenode = node_list[0]  # 起始节点
        nodelist_state = s  # 存储所有节点的消息列表
        s_ = []  # 下一个状态值
        penalty = 0  # 惩罚项
        acc_r = 0
        scheduled = False  # 表示本次调度中是否要进行真正的调度
        done = False  # 表示所有包是否输入完毕
        for node in node_list[1:-1]:  # 让网络中所有节点回到没有模拟调度之前
            index = node_list.index(node)
            if nodelist_state[index]:
                for m in nodelist_state[index]:  # 把存储的消息输入到中间节点中
                    acc_r = acc_r + 1/m.deadline
        # for node in node_list[1:]:  # 让网络中所有节点回到没有模拟调度之前
        #     index = node_list.index(node)
        #     node.messages.clear()
        #     if nodelist_state[index]:
        #         for m in nodelist_state[index]:  # 把存储的消息输入到中间节点中
        #             m.r_endtime = m.end_time
        #             node.messages.append(m)
        path = path_list[action]  # 获取选择的路径
        # print(action, len(path_list))
        message = sourcenode.messages.pop(0)
        message.path = path
        path_oneNode = path.nodes[0]  # 获取该路径的除了源节点的第一个节点
        path_oneNode.messages.append(message)  # 新消息进入选中的路径
        if count % 5 == 0:  # 此时进行一次调度
            scheduled = True
            for node in reversed(node_list[1:-1]):  # 遍历所有节点进行一次调度
                for down_node in node.down_nodes:
                    down_index = node.down_nodes.index(down_node)
                    weight = node.down_weights[down_index]
                    for i in range(weight):
                        if node.messages:
                            # for one_msg in node.messages:  # 对每个节点中的消息进行优先级排序
                            #     msg_nodeindex = one_msg.path.nodes.index(node)
                            #     steps = len(one_msg.path.nodes) - msg_nodeindex  # 表示该消息距离目的节点的跳数
                            #     one_msg.prior = one_msg.initial_prior - 0.3 * one_msg.end_time - 0.3 * steps
                            # node.messages.sort(key=lambda x: -x.prior)
                            for onemsg in node.messages:
                                msg_nodeindex = onemsg.path.nodes.index(node)
                                onemsg.nextnode = onemsg.path.nodes[msg_nodeindex + 1]
                                if onemsg.nextnode == down_node:
                                    node.messages.remove(onemsg)
                                    down_node.messages.append(onemsg)
                                    break
                                else:
                                    continue
            for node in node_list[1:-1]:  # 所有中间节点中的消息截止时间减1
                if node.messages:
                    for msg in node.messages:
                        msg.end_time = msg.end_time - 1
                        if msg.end_time < 0:
                            node.messages.remove(msg)
                            remove_count = remove_count + 1
                            penalty += 2
        for node in node_list[1:-1]:
            # 获取s_
            s_.append(len(node.messages))
        s_.append(message.end_time)
        # 获取r
        nodelist_state = [[0]]
        for node in node_list[1:]:
            index = node_list.index(node)
            nodelist_state.append([])
            if node.messages:
                for msg in node.messages:
                    nodelist_state[index].append(msg)
        flag = False
        r = - acc_r + 7 - penalty  # 7, 8.5, 10.1, 11.7, 13.4
        # r = 1.1 - penalty
        # r = Net.get_reward(self, flag, node_list, path, message, scheduled)
        endnode = node_list[-1]
        su_packets = len(endnode.messages)
        # 获取done
        if not sourcenode.messages:
            done = True
        return s_, r, done, nodelist_state, remove_count, acc_r, su_packets

    def get_reward(self, flag, node_list, path, message, scheduled):
        r = 1.0
        path_lastNode = path.nodes[-1]  # 获取该路径目的节点
        while not flag:  # 当该message还没有到达目的节点时
            for node in reversed(node_list[1:-1]):  # 遍历所有节点进行一次调度
                for down_node in node.down_nodes:
                    down_index = node.down_nodes.index(down_node)
                    weight = node.down_weights[down_index]
                    for i in range(weight):
                        if node.messages:
                            # for one_msg in node.messages:  # 对每个节点中的消息进行优先级排序
                            #     msg_nodeindex = one_msg.path.nodes.index(node)
                            #     steps = len(one_msg.path.nodes) - msg_nodeindex  # 表示该消息距离目的节点的跳数
                            #     one_msg.prior = one_msg.initial_prior - 0.3 * one_msg.r_endtime - 0.3 * steps
                            # node.messages.sort(key=lambda x: -x.prior)
                            for onemsg in node.messages:
                                msg_nodeindex = onemsg.path.nodes.index(node)
                                onemsg.nextnode = onemsg.path.nodes[msg_nodeindex + 1]
                                if onemsg.nextnode == down_node:
                                    node.messages.remove(onemsg)
                                    down_node.messages.append(onemsg)
                                    if message in path_lastNode.messages:
                                        r = 1.0
                                        flag = True
                                        return r
                                    break
                                else:
                                    continue
            for node in node_list[1:-1]:
                if node.messages:
                    for msg in node.messages:
                        if scheduled:
                            msg.r_endtime -= 1
                        msg.r_endtime -= 1
                        if message.r_endtime < 0:
                            r = 0.0
                            flag = True
                            return r
                        if msg.r_endtime < 0:
                            node.messages.remove(msg)
        return r


if __name__ == '__main__':
    net = Net()
    node_list, path_list = net.read_graph(net.node_list, net.path_list)
    print(node_list)
    print(path_list)
#     sys.setrecursionlimit(10000)  # 设置遍历深度上限
#     net = Net("test")
#     node_list, path_list = net.read_graph(net.node_list, net.path_list)
#     sourcenode = node_list[0]
#     net.get_data(sourcenode)
#     count = 0
#     remove_count = 0  # 记录丢弃的数据包的值
#     end_time = sourcenode.messages[0].end_time
#     s = []
#     states = [[0]]
#     ep_r = 0
#     print(len(path_list), len(node_list))
#     for node in node_list:
#         if node.down_nodes:
#             for down_node in node.down_nodes:
#                 print(node.name, down_node.name)
    for path in path_list:
        for node in path.nodes:
            print(node.name, end=" ")
        print()
#     for i in range(len(node_list) - 2):
#         s.append(0)
#         states.append([])
#     s.append(end_time)
#     states.append([])
#     while True:
#         count = count + 1
#         a = np.random.randint(0, len(path_list) - 1)
#         print(a)
#         s_, r, done, states, remove_count = net.schedule(a, count, states, node_list, path_list, remove_count)
#         s = s_
#         ep_r += r
#         print(count, r, ep_r, s_)


















