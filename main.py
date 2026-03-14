import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

engine = create_engine(
    "mysql+mysqlconnector://root@localhost:3306/ecommerce_db",
    connect_args={"password": "your_password"}
)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# SQL에서 데이터 불러오기
# ============================================================

country = pd.read_sql("""
    SELECT Country, COUNT(*) AS order_count
    FROM orders
    GROUP BY Country
    ORDER BY order_count DESC
    LIMIT 10
""", con=engine)

monthly = pd.read_sql("""
    SELECT 
        DATE_FORMAT(STR_TO_DATE(InvoiceDate, '%m/%d/%Y %H:%i'), '%Y-%m') AS month,
        ROUND(SUM(Quantity * UnitPrice), 2) AS revenue
    FROM orders
    WHERE Quantity > 0 AND UnitPrice > 0
    GROUP BY month
    ORDER BY month
""", con=engine)

products = pd.read_sql("""
    SELECT Description, SUM(Quantity) AS total_qty
    FROM orders
    WHERE Quantity > 0 AND UnitPrice > 0 AND Description IS NOT NULL
    GROUP BY Description
    ORDER BY total_qty DESC
    LIMIT 10
""", con=engine)

hourly = pd.read_sql("""
    SELECT 
        HOUR(STR_TO_DATE(InvoiceDate, '%m/%d/%Y %H:%i')) AS hour,
        COUNT(*) AS order_count
    FROM orders
    WHERE Quantity > 0
    GROUP BY hour
    ORDER BY hour
""", con=engine)

# ============================================================
# 시각화
# ============================================================

BLUE   = '#2563EB'
GREEN  = '#16A34A'
ORANGE = '#EA580C'
PURPLE = '#7C3AED'
LIGHT_ORANGE = '#FED7AA'

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('이커머스 데이터 분석 대시보드', fontsize=18, fontweight='bold')
fig.patch.set_facecolor('white')

# (1) 국가별 주문 수 → 파란색
ax1 = axes[0, 0]
ax1.barh(country['Country'][::-1], country['order_count'][::-1], color=BLUE)
for i, val in enumerate(country['order_count'][::-1]):
    ax1.text(val * 0.98, i, f'{val:,}', va='center', ha='right', color='white', fontsize=9)
ax1.set_title('국가별 주문 수 TOP 10', fontsize=13, fontweight='bold')
ax1.set_xlabel('주문 수')
ax1.grid(axis='x', alpha=0.3)

# (2) 월별 매출 추이 → 초록색
ax2 = axes[0, 1]
ax2.plot(monthly['month'], monthly['revenue'], marker='o', color=GREEN, linewidth=2)
ax2.fill_between(range(len(monthly)), monthly['revenue'], alpha=0.2, color=GREEN)
ax2.set_title('월별 매출 추이', fontsize=13, fontweight='bold')
ax2.set_xlabel('월')
ax2.set_ylabel('매출 (원)')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(alpha=0.3)

# (3) 인기 상품 TOP 10 → 주황색
ax3 = axes[1, 0]
ax3.barh(products['Description'][::-1], products['total_qty'][::-1], color=LIGHT_ORANGE,
         edgecolor=ORANGE, linewidth=0.5)
ax3.set_title('인기 상품 TOP 10', fontsize=13, fontweight='bold')
ax3.set_xlabel('판매 수량')
ax3.grid(axis='x', alpha=0.3)

# (4) 시간대별 주문 패턴 → 보라색
ax4 = axes[1, 1]
ax4.bar(hourly['hour'], hourly['order_count'], color=PURPLE, edgecolor='white')
ax4.set_title('시간대별 주문 패턴', fontsize=13, fontweight='bold')
ax4.set_xlabel('시간')
ax4.set_ylabel('주문 수')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('ecommerce_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 시각화 완료!")

# ============================================================
# RFM 고객 세분화
# ============================================================

rfm = pd.read_sql("""
    SELECT 
        CustomerID,
        DATEDIFF('2011-12-31', MAX(STR_TO_DATE(InvoiceDate, '%m/%d/%Y %H:%i'))) AS recency,
        COUNT(DISTINCT InvoiceNo) AS frequency,
        ROUND(SUM(Quantity * UnitPrice), 2) AS monetary
    FROM orders
    WHERE Quantity > 0 
      AND UnitPrice > 0
      AND CustomerID IS NOT NULL
    GROUP BY CustomerID
""", con=engine)

rfm['R_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1])
rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
rfm['M_score'] = pd.qcut(rfm['monetary'], 5, labels=[1,2,3,4,5])
rfm['RFM_score'] = rfm['R_score'].astype(int) + rfm['F_score'].astype(int) + rfm['M_score'].astype(int)

def classify(score):
    if score >= 13:
        return 'VIP'
    elif score >= 10:
        return '우수고객'
    elif score >= 7:
        return '일반고객'
    else:
        return '이탈위험'

rfm['segment'] = rfm['RFM_score'].apply(classify)
segment_counts = rfm['segment'].value_counts()
print("\n📊 고객 세분화 결과:")
print(segment_counts)

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle('고객 RFM 세분화 분석', fontsize=16, fontweight='bold')
fig3.patch.set_facecolor('white')

# RFM → 4가지 색상
colors = [BLUE, GREEN, ORANGE, PURPLE]

ax1 = axes3[0]
ax1.pie(segment_counts, labels=segment_counts.index,
        autopct='%1.1f%%', colors=colors, startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=2))
ax1.set_title('고객 등급 분포', fontsize=13, fontweight='bold')

ax2 = axes3[1]
segment_monetary = rfm.groupby('segment')['monetary'].mean().sort_values(ascending=True)
bars = ax2.barh(segment_monetary.index, segment_monetary.values,
                color=[BLUE, GREEN, ORANGE, PURPLE][:len(segment_monetary)])
for bar, val in zip(bars, segment_monetary.values):
    ax2.text(bar.get_width() * 0.98, bar.get_y() + bar.get_height()/2,
             f'{val:,.0f}원', va='center', ha='right', color='white', fontsize=10)
ax2.set_title('등급별 평균 구매금액', fontsize=13, fontweight='bold')
ax2.set_xlabel('평균 구매금액 (원)')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('ecommerce_rfm.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ RFM 분석 완료!")
