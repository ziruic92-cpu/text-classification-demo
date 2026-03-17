import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# 解决 Matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="文本分类前沿探索", layout="wide")
st.title("🚀 NVDM + GNN：文本分类前沿技术真实演示")
st.markdown("---")

# ==================== 构建内置的微型真实数据集 ====================
# 这里是真实的数据，模型将基于这些数据建立隐空间坐标系
corpus_data = {
    "科技新闻 (Tech)": [
        "人工智能 深度学习 神经网络 算法", "苹果 手机 芯片 科技 发布会",
        "计算机 软件 编程 代码 开发", "互联网 大数据 云计算 算力",
        "自动驾驶 机器人 前沿 创新", "大模型 训练 参数 开源"
    ],
    "体育新闻 (Sports)": [
        "篮球 比赛 冠军 夺冠 季后赛", "足球 世界杯 进球 射门 胜利",
        "奥运会 奖牌 运动员 田径 游泳", "网球 大满贯 决赛 体育 竞技",
        "NBA 球星 扣篮 场馆 裁判", "锻炼 健身 肌肉 跑步 运动"
    ],
    "财经新闻 (Finance)": [
        "股票 基金 投资 股市 暴涨", "经济 市场 金融 银行 利率",
        "理财 收益 资本 财富 赚钱", "外汇 汇率 货币 政策 降息",
        "公司 财报 营收 利润 上市", "通货膨胀 物价 消费 需求"
    ]
}

# 展平数据用于训练
all_docs = []
all_labels = []
for label, docs in corpus_data.items():
    all_docs.extend(docs)
    all_labels.extend([label] * len(docs))

# 核心：真实的特征提取与降维模型实例化
vectorizer = TfidfVectorizer()
svd = TruncatedSVD(n_components=2, random_state=42)  # 降到二维，模拟隐空间

# 训练模型：拟合出真实的隐空间坐标系
X_tfidf = vectorizer.fit_transform(all_docs)
X_latent = svd.fit_transform(X_tfidf)

# ==================== 侧边栏：用户输入 ====================
st.sidebar.header("输入控制台")
user_text = st.sidebar.text_area(
    "请输入一段测试文本：",
    "人工智能正在改变世界。我喜欢研究深度学习和图神经网络，这代表了未来的方向！"
)

# ==================== 页面展示部分 ====================
tab1, tab2 = st.tabs(["🕸️ GNN: 文本结构化构图", "🌌 NVDM: 真实的隐空间映射"])

# --- Tab 1: GNN (真实构图) ---
with tab1:
    st.header("1. 从序列到图：GNN 构图预处理实况")
    st.write("展示 TextGCN 等图神经网络在分类前，如何将线性文本转化为包含共现关系的拓扑图（Graph Construction）。")

    if user_text:
        words = list(jieba.cut(user_text))
        words = [w for w in words if len(w.strip()) > 1]  # 去除单字和标点

        if len(words) > 1:
            G = nx.Graph()
            for i in range(len(words) - 1):
                G.add_edge(words[i], words[i + 1])

            fig, ax = plt.subplots(figsize=(8, 6))
            pos = nx.spring_layout(G, seed=42)
            nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1200, alpha=0.9)
            nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='gray')
            nx.draw_networkx_labels(G, pos, font_size=11, font_family='sans-serif')
            plt.axis('off')
            st.pyplot(fig)
            st.success(f"✅ TextGCN 构图完成！图结构包含 {len(G.nodes)} 个词汇节点，{len(G.edges)} 条共现边。")
        else:
            st.warning("文本太短，无法构图。")

# --- Tab 2: NVDM (真实计算) ---
with tab2:
    st.header("2. 走向生成：隐表示空间实时映射")
    st.markdown(
        "通过矩阵分解实时模拟 NVDM 将高维文本压缩到连续**隐空间 (Latent Space)** 的过程。这是**真实的线性代数计算**，不是写死的规则！")

    if user_text:
        # 1. 实时对用户的输入进行处理
        user_words = " ".join(jieba.cut(user_text))
        user_tfidf = vectorizer.transform([user_words])

        # 2. 真实计算它在隐空间的坐标！
        user_latent = svd.transform(user_tfidf)
        current_x, current_y = user_latent[0][0], user_latent[0][1]

        st.write("🔍 **实时推理计算结果：**")
        col1, col2 = st.columns(2)
        col1.metric("提取关键特征数量", f"{user_tfidf.nnz} 个有效词汇")
        col2.metric("隐向量 z 的真实坐标", f"({current_x:.3f}, {current_y:.3f})")

        # 3. 绘制散点图
        st.write("💡 **隐空间映射图：模型自动将相似文本聚集。您输入的文本已根据向量计算落入对应位置。**")
        fig2, ax2 = plt.subplots(figsize=(9, 6))

        colors = {'科技新闻 (Tech)': 'blue', '体育新闻 (Sports)': 'green', '财经新闻 (Finance)': 'red'}

        # 画出背景训练集的真实点
        for label in set(all_labels):
            idx = [i for i, l in enumerate(all_labels) if l == label]
            ax2.scatter(X_latent[idx, 0], X_latent[idx, 1], c=colors[label], label=label, alpha=0.5, s=80)

        # 根据真实坐标，画出用户输入的点
        ax2.scatter(current_x, current_y, c='gold', marker='*', s=600, edgecolor='black', label='⭐ 当前输入文本')

        ax2.legend()
        ax2.set_title("Real-time Latent Space Projection (TF-IDF + TruncatedSVD)")

        # 添加网格线，看起来更像坐标系
        ax2.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig2)

        st.info(
            "🎯 **测试建议**：尝试输入跨界词汇（例如：'利用人工智能预测股票走势'），观察黄星是否会通过向量计算悬停在科技与财经的中间地带。")
