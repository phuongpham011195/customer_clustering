import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_option_menu import option_menu

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #155263ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <h3 style="text-align: center; font-size: 17px; color: white; margin-bottom: 5px;">
        TRƯỜNG ĐẠI HỌC KHOA HỌC TỰ NHIÊN<br>ĐHQG-HCM
    </h3>
    """,
    unsafe_allow_html=True
)

import streamlit as st

st.sidebar.markdown(
    """
    <h3 style="text-align: center; font-size: 18px; color: orange;">
        Đồ án tốt nghiệp Data Science<br>Project 1: Customer Segmentation
    </h3>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    page = option_menu(
        menu_title="Go to",  
        options=["Project Overview", "KMeans Model", "Manual Rule Based"],
        icons=["house", "bar-chart", "gear"],  
        default_index=0,
        styles={
            "container": {"background-color": "#ffc93cff"},
            "icon": {"color": "black", "font-size": "15px"}, 
            "nav-link": {"color": "black", "font-size": "15px", "text-align": "center"},
            "nav-link-selected": {"background-color": "#d2572eff", "color": "white", "font-size": "15px"},
        }
    )

st.sidebar.markdown(
    """
    <h3 style="text-align: center; font-size: 18px; color: white;">
        <br><br><br><br><br>Thành viên:<br><br>Phạm Mạch Lam Phương<br>Nguyễn Phạm Duy
    </h3>
    """,
    unsafe_allow_html=True
)

if page == "Project Overview":
    st.image("Grocieries.jpg")
    st.title("Project 1: Customer Segmentation")
    st.subheader("1. Đôi nét về Domain kinh doanh của khách hàng:")
    st.text("- Cửa hàng X chủ yếu bán các sản phẩm thiết yếu cho khách hàng như rau, củ, quả, thịt, cá, trứng, sữa, nước giải khát…")
    st.text("- Khách hàng của cửa hàng là khách hàng mua lẻ.")
    st.subheader("2. Mong muốn của khách hàng:")
    st.text("- Chủ cửa hàng X mong muốn có thể bán được nhiều hàng hóa hơn cũng như giới thiệu sản phẩm đến đúng đối tượng khách hàng, chăm sóc và làm hài lòng khách hàng.")
    st.text("=> Là một nhà phân tích dữ liệu, ta nghĩ đến bài toán phân cụm khách hàng (Clustering - Customer Segmentation) để phân cụm và có có chiến lược kinh doanh phù hợp đối với từng cụm khách hàng")
    st.image("cluster_analysis.png")
    st.subheader("3. Cách thức thực hiện:")
    st.markdown("""
    **Thử nghiệm xây dựng và đánh giá mô hình phân tích khách hàng RFM bằng các phương pháp sau:**
    - Phân cụm bằng thuật toán K-Means sử dụng Pyspark
    - Phân cụm bằng tập luật
    - Phân cụm bằng thuật toán K-Means và GMM bằng Python truyền thống
    - So sánh các mô hình bằng các chỉ số như Silhouette Score, Davies-Bouldin Index
    """)
    st.image("method.png")
    st.subheader("4. Đánh giá và lựa chọn thuật toán:")
    st.markdown("""
    **Sau khi phân tích và đánh giá dựa trên thời gian mà model xử lý và Silhouette Score, ta quyết định lựa chọn 2 phương pháp sau để giải quyết bài toán phân cụm khách hàng:**
    ** 1. Phân cụm bằng tập luật: Thời gian xử lý nhanh, phân cụm rõ ràng: Phân thành 8 nhóm khách hàng, phù hợp cho các chương trình khuyến mãi, các chiến lược kinh doanh có tính cá nhân hóa cao
    - 🎯 1. Marketing cá nhân hóa (Personalized Marketing)
    - 🛍 2. Đề xuất sản phẩm phù hợp hơn (Product Recommendation)
    - 🤝 3. Chăm sóc khách hàng tốt hơn
    - 💸 4. Tối ưu chi phí và nguồn lực
    """)
    st.image("RFM_Segments.png")
    st.markdown("""
    ** 2. Phân cụm bằng thuật toán K-Means bằng Python truyền thống: Thời gian xử lý nhanh, phân cụm rõ ràng: Phân thành 4 nhóm khách hàng, chiến lược kinh doanh sẽ mang tính tổng thể, dễ triển khai và quản lý hơn so với phân cụm chi tiết.
    - Tránh tình trạng overfitting
    - Ra quyết định nhanh, đơn giản
    """)
    st.image("Unsupervised_Segments.png")


elif page == "KMeans Model":
    st.image("Grocieries.jpg")
    st.title("🔍 Customer Segmentation by KMeans Model")
    # Open and read file to cosine_sim_new
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)

    # Giao diện nhập liệu
    st.title("Phân nhóm khách hàng bằng RFM")

    r = st.number_input("Recency (R)", min_value=0.0, step=1.0)
    f = st.number_input("Frequency (F)", min_value=0.0, step=1.0)
    m = st.number_input("Monetary (M)", min_value=0.0, step=1.0)

    if st.button("Dự đoán nhóm"):
        # Tạo array RFM
        user_rfm = np.array([[r, f, m]])
        
        # Dự đoán nhóm
        cluster = kmeans_model.predict(user_rfm)
        
        st.success(f"Khách hàng này thuộc nhóm: {cluster[0]}")

elif page == "Manual Rule Based":
    st.image("Grocieries.jpg")
    st.title("🔍 Customer Segmentation by Manual Rule Based")
    with open('manual_rule_based.pkl', 'rb') as f:
        manual_rule_based = pickle.load(f)

    # Giao diện nhập liệu
    st.title("Phân nhóm khách hàng bằng RFM")

    r = st.number_input("Recency (R)", min_value=0.0, step=1.0)
    f = st.number_input("Frequency (F)", min_value=0.0, step=1.0)
    m = st.number_input("Monetary (M)", min_value=0.0, step=1.0)

    if st.button("Dự đoán nhóm"):
        df_RFM = manual_rule_based['df_RFM']
        r_bins = manual_rule_based['r_bins']
        f_bins = manual_rule_based['f_bins']
        m_bins = manual_rule_based['m_bins']
        r_labels = manual_rule_based['r_labels']
        f_labels = manual_rule_based['f_labels']
        m_labels = manual_rule_based['m_labels']

        # --- Dữ liệu khách hàng mới ---
        new_customer = {
            'Recency': r,
            'Frequency': f,
            'Monetary': m
        }

        # --- Tính rank (so sánh với dữ liệu gốc) ---
        r_rank = (df_RFM['Recency'] < new_customer['Recency']).sum() + 1
        f_rank = (df_RFM['Frequency'] < new_customer['Frequency']).sum() + 1
        m_rank = (df_RFM['Monetary'] < new_customer['Monetary']).sum() + 1

        # --- Dự đoán R, F, M dựa vào phân vị đã lưu ---
        R = pd.cut([r_rank], bins=r_bins, labels=r_labels, include_lowest=True)[0]
        F = pd.cut([f_rank], bins=f_bins, labels=f_labels, include_lowest=True)[0]
        M = pd.cut([m_rank], bins=m_bins, labels=m_labels, include_lowest=True)[0]

        def a():
            if R + F + M == 12: # Khách hàng có số điểm tối đa là khách hàng VIP
                return 'VIPS'
            elif R == 4 and F == 1: # Khách hàng có ngày mua hàng gần nhất, tổng số giao dịch ít
                return 'NEW'
            elif R == 4 and F == 3 and M == 3: # Khách hàng thực hiện giao dịch gần đây, có phát sinh nhiều giao dịch, tổng giá trị giao dịch cao
                return 'LOYAL'
            elif R >= 3 and F == 4 and M == 4: # Khách hàng thực hiện giao dịch gần đây, có phát sinh nhiều giao dịch
                return 'ACTIVE'
            elif R == 1 and (R > 1 or M > 1): # Khách hàng đã lâu không mua hàng và tổng số tiền mua hàng, tổng số giao dịch trung bình
                return 'DORMANT'
            elif R == 1 and R == 1 and M == 1: # Khách hàng đã lâu không mua hàng và tổng số tiền mua hàng, tổng số giao dịch cũng rất thấp
                return 'CHURN'
            elif R == 2:
                return 'INACTIVE'
            else:
                return 'REGULARS'
        
        st.success(f"Khách hàng này thuộc nhóm: {a()}")