import json, textwrap, os
from pathlib import Path
import pandas as pd
from PIL import Image as PI
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image, Preformatted, KeepTogether, NextPageTemplate)

def resolve_font_path(name):
    project_root = Path(__file__).resolve().parent
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/truetype/liberation2") / name,
        project_root / ".venv" / "Lib" / "site-packages" / "matplotlib" / "mpl-data" / "fonts" / "ttf" / name,
        Path("C:/Windows/Fonts") / name,
        Path("C:/Program Files/Common Files/Microsoft Shared/Fonts") / name,
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None

normal_font = resolve_font_path("DejaVuSans.ttf") or resolve_font_path("DejaVuSansCondensed.ttf")
bold_font = resolve_font_path("DejaVuSans-Bold.ttf") or resolve_font_path("DejaVuSansCondensed-Bold.ttf")
mono_font = resolve_font_path("DejaVuSansMono.ttf") or resolve_font_path("DejaVuSansMono.ttf")
if normal_font:
    pdfmetrics.registerFont(TTFont("DV", normal_font))
if bold_font:
    pdfmetrics.registerFont(TTFont("DVB", bold_font))
if mono_font:
    pdfmetrics.registerFont(TTFont("DVM", mono_font))
if normal_font and bold_font:
    pdfmetrics.registerFontFamily("DV",normal="DV",bold="DVB",italic="DV",boldItalic="DVB")
FONT_NORMAL = "DV" if normal_font else "Helvetica"
FONT_BOLD = "DVB" if bold_font else "Helvetica-Bold"
FONT_MONO = "DVM" if mono_font else "Courier"
R=json.load(open("results.json")); df=pd.read_csv("data/ecommerce_sales_synthetic.csv")
NAVY=colors.HexColor("#1F3A8A"); LB=colors.HexColor("#EEF2F9"); GR=colors.HexColor("#6B7A99")
NAME,ROLL="Santhosh G","24AD102"; GH="https://github.com/santhosh07072007-pixel"
S=lambda n,**k: ParagraphStyle(n,**{**dict(fontName=FONT_NORMAL,fontSize=9,leading=13),**k})
body=S("b",alignment=TA_JUSTIFY,spaceAfter=5); small=S("s",fontSize=7.6,leading=10)
h1=S("h1",fontName=FONT_BOLD,fontSize=15,leading=19,textColor=NAVY,spaceBefore=4,spaceAfter=8)
h2=S("h2",fontName=FONT_BOLD,fontSize=10.5,leading=14,textColor=NAVY,spaceBefore=8,spaceAfter=4)
cap=S("c",fontName=FONT_BOLD,fontSize=8,leading=10,alignment=TA_CENTER,spaceAfter=6,spaceBefore=3)
tcap=S("tc",fontName=FONT_BOLD,fontSize=8.5,leading=11,spaceBefore=3,spaceAfter=3)
cell=S("cell",fontSize=7.8,leading=10); cellb=S("cellb",fontName=FONT_BOLD,fontSize=7.8,leading=10,textColor=colors.white)
P=lambda t,s=body: Paragraph(t,s)
def esc(t): return str(t).replace("&","&amp;")
def img(n,w=16.5*cm):
    iw,ih=PI.open(f"fig/{n}.png").size; return Image(f"fig/{n}.png",width=w,height=w*ih/iw)
def tbl(rows,widths,hdr=True,fs=7.8):
    data=[[Paragraph(esc(c),cellb if (hdr and i==0) else cell) for c in r] for i,r in enumerate(rows)]
    t=Table(data,colWidths=widths,repeatRows=1 if hdr else 0)
    st=[("GRID",(0,0),(-1,-1),.4,colors.HexColor("#C5CDE0")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]
    if hdr: st.append(("BACKGROUND",(0,0),(-1,0),NAVY))
    for i in range(1,len(rows)):
        if i%2==0: st.append(("BACKGROUND",(0,i),(-1,i),LB))
    t.setStyle(TableStyle(st)); return t
def pv(p): return "< 0.001" if p<1e-3 else f"{p:.4f}"
def sci(p): return f"{p:.2e}"
HDR="U21ADP05 – Exploratory Data Analysis and Visualization | Assignment II"
def later(c,d):
    c.saveState(); c.setFont(FONT_NORMAL,8); c.setFillColor(GR); c.drawRightString(A4[0]-2*cm,A4[1]-1.3*cm,HDR)
    c.setStrokeColor(colors.HexColor("#C5CDE0")); c.line(2*cm,A4[1]-1.5*cm,A4[0]-2*cm,A4[1]-1.5*cm)
    c.drawCentredString(A4[0]/2,1.1*cm,f"KPRIET – Dept. of AI & Data Science | Page {d.page-1}"); c.restoreState()
doc=BaseDocTemplate("E-commerce_Sales_Analysis_Case_Study_Report.pdf",pagesize=A4,leftMargin=2*cm,rightMargin=2*cm,topMargin=2.1*cm,bottomMargin=1.8*cm,
    title="E-commerce Sales Analysis – Case Study",author=f"{NAME} ({ROLL})")
fr=Frame(2*cm,1.8*cm,A4[0]-4*cm,A4[1]-3.9*cm,id="f")
doc.addPageTemplates([PageTemplate(id="cover",frames=[fr],onPage=lambda c,d:None),PageTemplate(id="body",frames=[fr],onPage=later)])
W=A4[0]-4*cm; st=[]
# ---------- derived numbers ----------
N=len(df); c0,c1=R["class_counts"]["0"],R["class_counts"]["1"]; m=R["mlp"]; l=R["lr"]; tu=R["tuning"]; bk=R["best_cfg"]
maj=max(R["split"]["test"][1:])/R["split"]["test"][0]*100
rt=R["rev_total"]; ec=R["rev_by_cat"]["Electronics"]; nd=R["rev_by_month"]["11"]+R["rev_by_month"]["12"]
# ---------- Cover ----------
st+= [Spacer(1,.3*cm),P("<b>KPR INSTITUTE OF ENGINEERING AND TECHNOLOGY</b>",S("x",fontName=FONT_BOLD,fontSize=14,leading=18,alignment=TA_CENTER,textColor=NAVY)),
 P("(Autonomous, Affiliated to Anna University, Accredited by NAAC 'A' Grade)",S("x2",alignment=TA_CENTER,fontSize=8.5)),Spacer(1,.2*cm),
 P("Department of Artificial Intelligence and Data Science",S("x3",fontName=FONT_BOLD,alignment=TA_CENTER,fontSize=11,leading=14)),Spacer(1,1*cm),
 P("CASE STUDY",S("x4",fontName=FONT_BOLD,alignment=TA_CENTER,fontSize=20,leading=24,textColor=NAVY)),Spacer(1,.4*cm),
 P("E-commerce Sales Analysis:<br/>Exploratory Data Analysis, Statistical Visualization,<br/>and Deep Neural Network Prediction of Repeat Purchases",S("x5",fontName=FONT_BOLD,alignment=TA_CENTER,fontSize=14,leading=19)),Spacer(1,1*cm)]
cover=[["Course Code &amp; Title","U21ADP05 – Exploratory Data Analysis and Visualization"],["Assignment","Assignment II"],
 ["Academic Year &amp; Semester","2026 – 2027 (ODD SEM) | Semester V"],["Batch","2024 · 2028"],["Course Coordinator","Ms. REVATHY M"],
 ["Student Name",NAME],["Roll / Register Number",ROLL],["Date of Submission","15 / 10 / 2026"],
 ["Evaluation Rubrics","Total 40 Marks (CO3: 8 | CO4: 16 | CO5: 16 + Deep Learning Enrichment Component)"],["GitHub Repository",GH]]
t=Table([[Paragraph(a,S("k",fontName=FONT_BOLD,fontSize=9.5,leading=13)),Paragraph(esc(b).replace("&amp;amp;","&amp;"),S("v",fontSize=9.5,leading=13))] for a,b in cover],colWidths=[5.2*cm,W-5.2*cm])
t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.5,colors.HexColor("#C5CDE0")),("BACKGROUND",(0,0),(0,-1),LB),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
st+=[t,NextPageTemplate("body"),PageBreak()]
# ---------- 2 Abstract ----------
abstract=(f"Online retailers depend on repeat customers, yet identifying who will return is difficult. This study analyses a synthetic e-commerce dataset of {N:,} orders "
 f"with {R['n_features']-1} customer, product and delivery features, following Ben Fry's seven stages of data visualization. Preprocessing covers duplicate audit, Tukey IQR outlier quantification, "
 f"leakage-free imputation and stratified 70/15/15 partitioning. Welch t-tests, Mann-Whitney U tests and Chi-Square tests show that Prime membership, faster shipping and prior order history are linked to repeat purchase. "
 f"A multi-layer perceptron, tuned over three configurations with early stopping, reaches {m['acc']*100:.2f}% test accuracy and ROC-AUC {m['auc']:.4f}, "
 f"close to a logistic regression baseline ({l['auc']:.4f}). Electronics yields {ec/rt*100:.1f}% of revenue, and November and December contribute {nd/rt*100:.1f}%.")
wc=len(abstract.split()); assert 100<=wc<=150,wc
st+=[P("2. Abstract",h1),P(abstract),P(f"<i>(Word Count: {wc} words — within the required 100–150 word specification.)</i>",small),Spacer(1,.3*cm),
 P("<b>Data note.</b> No real transaction log was supplied, so the analysis runs on a synthetic dataset generated with a fixed random seed (42) by the script in Section 11. Every number, table and figure in this report comes from that run. The conclusions describe the synthetic data and should not be read as findings about a real retailer.",body)]
# ---------- 3 Intro ----------
st+=[PageBreak(),P("3. Introduction &amp; Objective",h1),P("3.1 Business Background and Motivation",h2),
 P("Acquiring a new online customer typically costs far more than retaining an existing one, so repeat purchase behaviour is a central metric for e-commerce planning. Revenue in online retail is also unevenly distributed: a small number of categories, seasons and large baskets usually account for a disproportionate share of sales. Understanding both patterns lets a retailer decide where to stock, when to discount and which customers to target with loyalty offers."),
 P("When a retailer cannot tell which customers are likely to return, it faces three recurring problems:"),
 P("1. <b>Misallocated discounts:</b> blanket promotions erode margin on customers who would have returned anyway."),
 P("2. <b>Unplanned seasonal peaks:</b> demand spikes in festive months strain inventory and delivery capacity."),
 P("3. <b>Delivery-driven churn:</b> slow shipping lowers ratings and weakens the chance of a second order."),
 P("This case study turns these questions into a measurable pipeline: explore revenue structure, test which order and customer attributes differ between one-time and repeat buyers, and train a neural network to estimate the probability of a repeat purchase."),
 P("3.2 Technical Objectives",h2),
 P(f"<b>Step 1 — Dataset Selection:</b> Generate a reproducible synthetic e-commerce orders dataset of {N:,} records and {R['n_features']-1} features (15 original order, customer and delivery attributes plus 5 engineered features, strictly exceeding the ≥ 15 feature requirement)."),
 P("<b>Step 2 — Data Understanding &amp; Preprocessing:</b> Inspect missing values, verify duplicate independence, compute Tukey IQR outlier bounds, encode categorical variables, standardize numeric scales and enforce stratified 70/15/15 splits without data leakage."),
 P("<b>Step 3 — Exploratory Data Visualization:</b> Demonstrate Ben Fry's seven stages (Acquire, Parse, Filter, Mine, Represent, Refine, Interact) through figures covering univariate, categorical, bivariate, temporal and multivariate views."),
 P("<b>Step 4 — Statistical &amp; Distributional Exploration (CO4):</b> Describe skewness and kurtosis, and run Welch t-tests, Mann-Whitney U tests, Chi-Square tests of independence and Cramér's V."),
 P("<b>Step 5 — Deep Learning Neural Modeling (CO5 &amp; Enrichment):</b> Build, regularize and tune a multi-layer perceptron (MLP) across three hyperparameter configurations using scikit-learn, and evaluate convergence, generalization and permutation feature importance."),
 P("<b>Step 6 &amp; 7 — Model Evaluation &amp; Synthesis:</b> Produce the loss curve, accuracy curve, confusion matrix and ROC curve, benchmark against logistic regression, and summarize findings with references.")]
# ---------- 4 Dataset ----------
def rng_(c,f="{:.0f}"): return f"{f.format(df[c].min())} to {f.format(df[c].max())}"
st+=[PageBreak(),P("4. Dataset Description",h1),
 P(f"The analysis uses a synthetic orders table of {N:,} rows. Each row is one order placed by a customer, described by demographic, product, payment, delivery and engagement attributes. The target variable <b>repeat_purchase</b> equals 1 if the customer placed a further order after this one and 0 otherwise. Category-level base prices, seasonal discount uplift in November and December, longer delivery in the Central region and Prime-member effects were built into the generator so that realistic patterns exist to be discovered; the generator code is shown in Section 11."),
 P("4.1 Feature Schema Dictionary",h2),P("Table 1: Feature schema of the e-commerce orders dataset",tcap)]
sch=[["Field Name","Data Type","Domain / Values","Category","Operational Description"],
["order_month","Categorical (int)","1 to 12","Temporal","Calendar month of the order"],
["customer_age","Continuous",rng_("customer_age")+" years","Demographic","Age of the customer"],
["gender","Categorical","F, M","Demographic","Customer gender"],
["region","Categorical","North, South, East, West, Central","Geographic","Delivery region"],
["category","Categorical","Electronics, Fashion, Home & Kitchen, Beauty, Books, Sports","Product","Product category"],
["payment_method","Categorical","Credit Card, UPI, Debit Card, COD, Wallet","Transactional","Payment mode used"],
["device","Categorical","Mobile, Desktop, Tablet","Behavioural","Device used to place the order"],
["unit_price","Continuous",rng_("unit_price","{:.2f}")+" currency units","Product","Price of one unit"],
["quantity","Discrete",rng_("quantity")+" units","Transactional","Units in the order"],
["discount_pct","Continuous",rng_("discount_pct","{:.1f}")+" %","Transactional","Discount applied to the order"],
["shipping_days","Discrete",rng_("shipping_days")+" days","Logistics","Days from order to delivery"],
["is_prime","Binary Int","0 (No), 1 (Yes)","Customer","Prime membership indicator"],
["prior_orders","Discrete",rng_("prior_orders")+" orders","Customer","Orders placed before this one"],
["session_minutes","Continuous",rng_("session_minutes","{:.1f}")+" min","Behavioural","Time spent on site before purchase"],
["customer_rating","Continuous",rng_("customer_rating","{:.1f}")+" stars","Customer","Delivery and product rating"],
["order_value","Float (Engineered)",rng_("order_value","{:.2f}"),"Engineered","unit_price × quantity × (1 − discount)"],
["discount_amount","Float (Engineered)",rng_("discount_amount","{:.2f}"),"Engineered","Absolute monetary discount given"],
["value_per_session_min","Float (Engineered)",rng_("value_per_session_min","{:.2f}"),"Engineered","Order value per minute of browsing"],
["late_delivery_flag","Binary (Engineered)","0 (&lt; 7 days), 1 (≥ 7 days)","Engineered","Delivery took a week or longer"],
["loyalty_score","Integer (Engineered)",rng_("loyalty_score")+" points","Engineered","2×prior orders + 3×Prime + 2×(rating ≥ 4)"],
["target","Binary Int","0 (One-time), 1 (Repeat)","Target","Customer made a repeat purchase"]]
t=Table([[Paragraph(c.replace("&","&amp;").replace("&amp;lt;","&lt;"),cellb if i==0 else cell) for c in r] for i,r in enumerate(sch)],colWidths=[3.2*cm,2.5*cm,4.5*cm,2.2*cm,W-12.4*cm],repeatRows=1)
ts=[("GRID",(0,0),(-1,-1),.4,colors.HexColor("#C5CDE0")),("BACKGROUND",(0,0),(-1,0),NAVY),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),2.5),("BOTTOMPADDING",(0,0),(-1,-1),2.5)]
ts+=[("BACKGROUND",(0,i),(-1,i),LB) for i in range(2,len(sch),2)]; t.setStyle(TableStyle(ts)); st+=[t]
st+=[PageBreak(),P("4. Dataset Description (Cont.)",h1),P("4.2 Descriptive Summary Statistics of Continuous Variables",h2),P(f"Table 2: Descriptive statistics (n = {N:,}; missing values excluded)",tcap)]
dd={d["index"]:d for d in R["desc"]}; vs=["customer_age","unit_price","quantity","discount_pct","shipping_days","order_value","session_minutes","customer_rating"]
lab=["age","unit_price","quantity","discount %","ship days","order_value","session min","rating"]
rows=[["Statistic"]+lab]
for k,n,f in [("count","Count","{:.0f}"),("mean","Mean","{:.2f}"),("std","Std Dev","{:.2f}"),("50%","Median","{:.2f}"),("min","Min","{:.2f}"),("25%","Q1 (25%)","{:.2f}"),("75%","Q3 (75%)","{:.2f}"),("max","Max","{:.2f}"),("skew","Skewness","{:+.2f}"),("kurt","Kurtosis","{:+.2f}")]:
    rows.append([n]+[f.format(dd[v][k]) for v in vs])
st+=[tbl(rows,[2.2*cm]+[(W-2.2*cm)/8]*8),Spacer(1,.2*cm),P("Statistical Distribution Insights",h2),
 P(f"<b>1. Customer age</b> ranges from {dd['customer_age']['min']:.0f} to {dd['customer_age']['max']:.0f} years (mean {dd['customer_age']['mean']:.2f}, median {dd['customer_age']['50%']:.0f}) and is nearly symmetric (skewness {dd['customer_age']['skew']:+.2f}), describing a young-adult shopper base."),
 P(f"<b>2. Price and order value</b> are strongly right-skewed. Unit price has skewness {dd['unit_price']['skew']:+.2f} and order value {dd['order_value']['skew']:+.2f} with kurtosis {dd['order_value']['kurt']:+.2f}: the median order is {dd['order_value']['50%']:.2f} against a mean of {dd['order_value']['mean']:.2f}, because a small number of high-priced Electronics baskets (maximum {dd['order_value']['max']:.2f}) pull the mean upward. Median-based summaries and log scales are therefore more representative than the mean."),
 P(f"<b>3. Delivery and engagement.</b> Shipping time has median {dd['shipping_days']['50%']:.0f} days with a long tail to {dd['shipping_days']['max']:.0f} days (skewness {dd['shipping_days']['skew']:+.2f}). Session length is also right-skewed ({dd['session_minutes']['skew']:+.2f}). Customer ratings are left-skewed ({dd['customer_rating']['skew']:+.2f}) with mean {dd['customer_rating']['mean']:.2f}, so most customers are satisfied and low ratings form the minority tail.")]
# ---------- 5 EDA ----------
o=R["outliers"]; sp=R["split"]
st+=[PageBreak(),P("5. EDA and Preprocessing",h1),P("5.1 Preprocessing Pipeline Execution",h2),
 P(f"<b>1. Whitespace &amp; Missing Value Treatment:</b> Missing values occur in two attributes: customer_rating ({R['missing']['customer_rating']} values, {R['missing']['customer_rating']/N*100:.2f}%) and session_minutes ({R['missing']['session_minutes']} values, {R['missing']['session_minutes']/N*100:.2f}%). To avoid leakage, the data were split first and imputation used the training-partition median only (customer_rating = {R['impute']['customer_rating']}, session_minutes = {R['impute']['session_minutes']}). No orders were discarded."),
 P(f"<b>2. Duplicate Record Audit:</b> An audit across all columns except the order identifier found {R['dups']} duplicate rows."),
 P("<b>3. Outlier Detection via Interquartile Range (IQR):</b>")]
for c,lab_ in [("unit_price","unit_price"),("order_value","order_value"),("shipping_days","shipping_days"),("session_minutes","session_minutes"),("discount_pct","discount_pct")]:
    d=o[c]; st.append(P(f"• <b>{lab_}:</b> IQR = {d['iqr']}, lower bound = {d['lo']}, upper bound = {d['hi']}. Detected outliers = {d['n']} ({d['n']/N*100:.2f}%), maximum {d['mx']:.2f}.",S("bl",fontSize=9,leading=12.5,leftIndent=14,spaceAfter=2)))
st+=[P("<b>Business assessment:</b> These outliers are genuine high-value purchases (premium electronics, multi-unit baskets), long cross-region deliveries and long browsing sessions rather than data-entry errors. They were retained, and standardization together with a log-scaled view in the figures keeps them from dominating the analysis."),
 P(f"<b>4. Categorical Encoding:</b> Binary attributes (is_prime, late_delivery_flag) were kept as {{0, 1}}. Nominal variables (gender, region, category, payment_method, device) were one-hot encoded with drop_first=True, expanding the feature space to <b>{R['input_dim']} numeric input dimensions</b>."),
 P("<b>5. Feature Standardization:</b> The 13 continuous and count variables were standardized with StandardScaler (mean 0, standard deviation 1), fitted on the training partition and applied unchanged to the validation and test partitions."),
 P(f"<b>6. Stratified Splitting:</b> Orders were partitioned as shown in Table 3. Stratification keeps the repeat-purchase share near {c1/N*100:.1f}% in every subset."),
 P("Table 3: Stratified train / validation / test partitions",tcap)]
rows=[["Partition","Share","Rows","One-time (0)","Repeat (1)"]]
for k,n,s in [("train","Training","70%"),("val","Validation","15%"),("test","Test","15%")]:
    a,b,c_=sp[k]; rows.append([n,s,str(a),f"{b} ({b/a*100:.1f}%)",f"{c_} ({c_/a*100:.1f}%)"])
st+=[tbl(rows,[3.5*cm,2.5*cm,2.5*cm,(W-8.5*cm)/2,(W-8.5*cm)/2])]
# ---------- 6 Viz ----------
st+=[PageBreak(),P("6. Data Visualization (Ben Fry's 7 Stages + 6 Exploratory Figures)",h1),P("6.1 Ben Fry's 7 Stages of Data Visualization Framework",h2),P("Table 4: Mapping of Fry's seven stages to this case study",tcap)]
fry=[["Stage","Operational Implementation in this Case Study"],
["1. Acquire","Generated a reproducible synthetic orders table (seed 42) into a Pandas DataFrame and saved it as ecommerce_sales_synthetic.csv."],
["2. Parse","Typed each field as continuous, discrete, binary or categorical, and defined the target (repeat_purchase → target)."],
["3. Filter","Located missing values in customer_rating and session_minutes, verified duplicate independence, and flagged outliers with Tukey IQR fences."],
["4. Mine","Ran Welch t-tests, Mann-Whitney U tests, Chi-Square tests with Cramér's V, and Pearson correlations."],
["5. Represent","Chose encodings suited to each variable: bar and donut-style summaries, log-scaled histograms with KDE, violin and box plots, heatmaps, line and regression scatter plots."],
["6. Refine","Applied a consistent two-colour palette (#2B5F91 for one-time, #D94A38 for repeat), annotated values and exported figures at 200 DPI."],
["7. Interact","Interaction here is limited to the saved figures, CSV and results.json, which can be re-explored by re-running the script; an interactive dashboard is listed as future work."]]
st+=[tbl(fry,[3*cm,W-3*cm]),Spacer(1,.3*cm),P("6.2 Visualization 1: Target Distribution and Revenue by Category",h2),img("f1"),
 P("Figure 1: Repeat-purchase class distribution (left) and total revenue by product category (right)",cap),
 P(f"<b>What it shows:</b> The count of one-time ({c0}; {c0/N*100:.1f}%) and repeat ({c1}; {c1/N*100:.1f}%) orders, and total revenue by category."),
 P("<b>Why it was created:</b> To establish class priors before modelling and to see where revenue is concentrated."),
 P(f"<b>Insights obtained:</b> The class ratio is about {c1/c0:.2f}:1 in favour of repeat purchase, so accuracy alone must be read against a majority-class baseline of {maj:.1f}%. Electronics produces {ec:,.0f} of {rt:,.0f} total revenue ({ec/rt*100:.1f}%) while Books contributes only {R['rev_by_cat']['Books']/rt*100:.1f}%.")]
tv=R["ttests"]
st+=[PageBreak(),P("6. Data Visualization (Cont.)",h1),P("6.3 Visualization 2: Continuous Probability Distributions (KDE &amp; Histograms)",h2),img("f2"),
 P("Figure 2: Log-scaled histogram with KDE and violin plot of order value by repeat status",cap),
 P("<b>What it shows:</b> Empirical density, KDE curves and quartile spread of order value for one-time and repeat customers."),
 P("<b>Why it was created:</b> To check whether larger baskets are associated with returning, and to handle the heavy right tail with a log axis."),
 P(f"<b>Insights obtained:</b> Order value is roughly log-normal. Repeat customers have a lower mean order value ({tv['order_value']['m1']:.2f}) than one-time customers ({tv['order_value']['m0']:.2f}); Welch t = {tv['order_value']['t']}, p = {sci(tv['order_value']['p'])}, and Mann-Whitney p = {sci(tv['order_value']['mwp'])}. Expensive Electronics baskets are bought less often by returning customers."),
 P("6.4 Visualization 3: Bivariate Boxplots with Outlier Bounds",h2),img("f3",15*cm),
 P("Figure 3: Multi-panel boxplots with Tukey IQR upper fences for order value, shipping days, discount and session length",cap),
 P("<b>What it shows:</b> Medians, interquartile ranges and outlier fences for four numeric variables grouped by repeat status."),
 P(f"<b>Insights obtained:</b> Shipping time shows the clearest separation: repeat customers have a median of {tv['shipping_days']['med1']:.0f} days versus {tv['shipping_days']['med0']:.0f} for one-time customers (Welch t = {tv['shipping_days']['t']}, p = {sci(tv['shipping_days']['p'])}). Discount differs modestly ({tv['discount_pct']['m1']:.2f}% vs {tv['discount_pct']['m0']:.2f}%, p = {tv['discount_pct']['p']:.4f}), whereas session length does not (p = {tv['session_minutes']['p']:.2f}).")]
ch=R["chi"]
st+=[PageBreak(),P("6. Data Visualization (Cont.)",h1),P("6.5 Visualization 4: Customer Segments vs Repeat Rate (Chi-Square Tests)",h2),img("f4"),
 P("Figure 4: Repeat-purchase rate by product category, payment method and Prime membership",cap),
 P("<b>What it shows:</b> The percentage of orders followed by a repeat purchase within each category, payment method and Prime group."),
 P("<b>Why it was created:</b> To find which categorical segments carry the strongest link to retention."),
 P(f"<b>Insights obtained:</b> Prime members repeat at {R['rate_prime']['1']}% versus {R['rate_prime']['0']}% for non-members (χ² = {ch['is_prime']['chi2']}, p {pv(ch['is_prime']['p'])}, Cramér's V = {ch['is_prime']['v']}). Category is also significant (χ² = {ch['category']['chi2']}, dof = {ch['category']['dof']}, p = {ch['category']['p']:.4f}, V = {ch['category']['v']}), with Beauty ({R['rate_by_cat']['Beauty']}%) and Books ({R['rate_by_cat']['Books']}%) highest and Electronics ({R['rate_by_cat']['Electronics']}%) lowest. Payment method (p = {ch['payment_method']['p']:.3f}), region (p = {ch['region']['p']:.3f}) and device (p = {ch['device']['p']:.3f}) are not statistically significant at the 5% level, although cash-on-delivery has the lowest repeat rate ({R['rate_by_pay']['COD']}%)."),
 P("6.6 Visualization 5: Multivariate Pearson Correlation Matrix",h2),img("f5",11.5*cm),
 P("Figure 5: Lower-triangular Pearson correlation heatmap of order, customer and delivery variables with the target",cap),
 P(f"<b>Insights obtained:</b> Linear associations with repeat purchase are modest. The strongest are customer_rating ({R['corr_target']['customer_rating']:+.2f}), prior_orders ({R['corr_target']['prior_orders']:+.2f}), is_prime ({R['corr_target']['is_prime']:+.2f}) and shipping_days ({R['corr_target']['shipping_days']:+.2f}). Order value is weakly negative ({R['corr_target']['order_value']:+.2f}). Correlation among predictors is low to moderate, so no variable was dropped.")]
st+=[PageBreak(),P("6. Data Visualization (Cont.)",h1),P("6.7 Visualization 6: Seasonality and Delivery Delay vs Rating",h2),img("f6"),
 P("Figure 6: Monthly revenue seasonality (left) and shipping delay vs customer rating with regression lines (right)",cap),
 P("<b>What it shows:</b> Revenue summed by calendar month, and the relationship between delivery time and customer rating for each customer group."),
 P("<b>Why it was created:</b> To quantify seasonal demand and to show how delivery performance feeds customer satisfaction."),
 P(f"<b>Insights obtained:</b> Revenue climbs sharply in the festive months: November ({R['rev_by_month']['11']:,.0f}) and December ({R['rev_by_month']['12']:,.0f}) together contribute {nd/rt*100:.1f}% of annual revenue, against 16.7% expected if revenue were spread evenly across months. Shipping days and rating are moderately negatively correlated (Pearson r = {R['ship_rating_r'][0]}, p &lt; 0.001): every extra day of delivery is associated with lower satisfaction, which in turn relates to lower repeat probability."),
 P(f"<b>Regional view:</b> South ({R['rev_by_region']['South']:,.0f}) and North ({R['rev_by_region']['North']:,.0f}) lead revenue, while Central ({R['rev_by_region']['Central']:,.0f}) trails and also has the longest delivery times by construction of the generator.")]
# ---------- 7 DL ----------
b=tu[bk]; hh=b["h"]
st+=[PageBreak(),P("7. Deep Learning Model Building &amp; Hyperparameter Tuning",h1),P("7.1 Neural Network Formulation &amp; Architecture",h2),
 P(f"The classifier is a feed-forward multi-layer perceptron implemented with scikit-learn's MLPClassifier (TensorFlow was not available in the execution environment). It maps a standardized input vector x in R<super>{R['input_dim']}</super> to a probability of repeat purchase:"),
 P("ŷ = σ( W<sub>3</sub> · ReLU( W<sub>2</sub> · ReLU( W<sub>1</sub> x + b<sub>1</sub> ) + b<sub>2</sub> ) + b<sub>3</sub> )",S("eq",alignment=TA_CENTER,fontSize=10,leading=14,spaceAfter=6)),
 P("where σ(z) = 1 / (1 + e<super>−z</super>). Training minimizes binary cross-entropy with an L2 weight penalty (parameter alpha), optimized by Adam with mini-batches of 32. scikit-learn's MLP has no dropout layer, so L2 regularization and early stopping provide the regularization."),
 P(f"Table 5: MLP layer specification for the selected configuration ({bk.split(' (')[0]})",tcap)]
rows=[["Layer","Input Dims","Output Dims","Activation / Regularization","Operational Role"],
 ["Input Layer",f"{R['input_dim']} features",f"{hh[0]} units","Affine linear + ReLU","Projects scaled numeric and dummy features into a latent space"],
 ["Hidden Layer 2",f"{hh[0]} units",f"{hh[1]} units","Dense + ReLU","Condenses interactions into a compact representation"],
 ["Output Layer",f"{hh[1]} units","1 unit","Sigmoid, log-loss","Probability of repeat purchase P(y=1 | x)"],
 ["Regularization","—","—",f"L2 alpha = {b['alpha']}; early stopping (patience 15)","Limits overfitting on 1,400 training rows"]]
st+=[tbl(rows,[2.6*cm,2.4*cm,2.3*cm,4.2*cm,W-11.5*cm]),P("7.2 Hyperparameter Tuning Experiment",h2),
 P("Three candidate architectures were trained on the training partition and compared on the validation partition. Each run used manual epoch-by-epoch training with early stopping (patience = 15 epochs) and restoration of the weights from the epoch with the lowest validation loss."),
 P("Table 6: Hyperparameter grid and validation results",tcap)]
rows=[["Configuration","Hidden Layers","Learning Rate","L2 alpha","Best Epoch","Best Val Loss","Val ROC-AUC","Params"]]
for k,v in tu.items(): rows.append([k+(" ★" if k==bk else ""),"("+", ".join(map(str,v["h"]))+")",str(v["lr"]),str(v["alpha"]),str(v["ep"]),f"{v['loss']:.4f}",f"{v['auc']:.4f}",f"{v['np']:,}"])
st+=[tbl(rows,[3.6*cm,2*cm,1.9*cm,1.5*cm,1.5*cm,1.9*cm,1.9*cm,W-14.3*cm]),
 P(f"<b>Selection Rationale:</b> {bk} was selected because it achieved the lowest validation cross-entropy ({b['loss']:.4f}) and the highest validation ROC-AUC ({b['auc']:.4f}). The three configurations differ only slightly (validation loss between {min(v['loss'] for v in tu.values()):.4f} and {max(v['loss'] for v in tu.values()):.4f}), so the choice is not strongly decisive; with 300 validation rows, such small gaps lie within sampling noise. The selected model is also the largest ({b['np']:,} trainable parameters).")]
st+=[PageBreak(),P("7. Deep Learning Model Building (Cont.)",h1),P("7.3 Permutation Feature Importance Analysis",h2),
 P(f"Model-agnostic permutation importance was computed on the held-out test partition (N = {sp['test'][0]}). Each feature is shuffled 10 times and the mean drop in test ROC-AUC is recorded:"),P("Table 7: Top-10 permutation feature importances",tcap)]
rows=[["Rank","Feature Name","Test ROC-AUC Drop (Importance)","Standard Deviation"]]
for i,(n,mn,sd) in enumerate(R["perm"],1): rows.append([f"#{i}",n,f"{mn:.5f}",f"± {sd:.5f}"])
pm={n:mn for n,mn,_ in R["perm"]}
st+=[tbl(rows,[1.5*cm,5.5*cm,5.5*cm,W-12.5*cm]),Spacer(1,.2*cm),P("Interpretability &amp; Synthesis",h2),
 P(f"<b>1. Loyalty drives the prediction.</b> The engineered loyalty_score (built from prior orders, Prime status and high rating) is the top feature (AUC drop {R['perm'][0][1]:.4f}), followed by prior_orders ({R['perm'][1][1]:.4f}). Customers with an order history are the most likely to come back."),
 P(f"<b>2. Delivery experience matters.</b> shipping_days ranks third and customer_rating fourth, consistent with the statistical tests in Section 6: slow delivery and poor ratings reduce repeat purchase."),
 P("<b>3. Caution on interpretation.</b> loyalty_score overlaps with prior_orders, is_prime and customer_rating, so permuting one of them can be partly compensated by the others; importance is shared across correlated features. All importances are small (the largest drop is under 0.04 AUC) and several have standard deviations of similar size, so only the top few rankings should be treated as reliable.")]
# ---------- 8 Results ----------
H=R
st+=[PageBreak(),P("8. Result Visualization &amp; Interpretation (Step 5 Evaluation)",h1),P("8.1 Figure 7: Training &amp; Validation Loss vs Epoch Trajectory",h2),img("f7",14.5*cm),
 P("Figure 7: Training and validation binary cross-entropy loss vs epoch with optimal checkpoint",cap),
 P(f"<b>Chart Description:</b> Training and validation log-loss for the selected configuration, with the lowest-validation-loss epoch marked."),
 P(f"<b>Performance Interpretation:</b> Training loss falls steadily from {H['train_loss0']} to {H['train_loss_end']} over {H['epochs_run']} epochs. Validation loss reaches its minimum of {H['best_val_loss']:.4f} at epoch {H['best_epoch']} and then rises, so early stopping triggered and the epoch-{H['best_epoch']} weights were restored. The widening gap shows the network begins to memorize the 1,400 training orders once the generalizable signal has been learned, which is expected for a dataset of this size and signal strength.")]
st+=[PageBreak(),P("8. Result Visualization &amp; Interpretation (Cont.)",h1),P("8.2 Figure 8: Training &amp; Validation Accuracy vs Epoch Trajectory",h2),img("f8",14.5*cm),
 P("Figure 8: Training and validation accuracy across learning epochs",cap),
 P(f"<b>Performance Interpretation:</b> Training accuracy rises to {H['train_acc_end']*100:.1f}% while validation accuracy stays near {H['best_val_acc']*100:.0f}%. The validation figure is only a few points above the majority-class rate of {maj:.1f}%, so most of the model's apparent skill lies in ranking customers (see ROC-AUC) rather than in hard classification at a 0.5 cutoff."),
 P("8.3 Figure 9: Out-of-Sample Test Confusion Matrix",h2),img("f9",9.5*cm),P(f"Figure 9: Confusion matrix on the held-out test set (n = {sp['test'][0]})",cap),
 P(f"<b>Performance Interpretation:</b> At the default 0.5 threshold the model correctly identifies <b>{m['tp']} repeat buyers</b> (recall {m['rec']*100:.1f}%) and {m['tn']} one-time buyers (specificity {m['spec']*100:.1f}%). It misses {m['fn']} repeat customers (false negatives) and wrongly flags {m['fp']} one-time customers as repeat (false positives). Because repeat customers are the majority class, the model leans toward predicting repeat purchase.")]
y=R["mlp_youden"]
st+=[PageBreak(),P("8. Result Visualization &amp; Interpretation (Cont.)",h1),P("8.4 Figure 10: Receiver Operating Characteristic (ROC) &amp; AUC Comparison",h2),img("f10",11*cm),
 P("Figure 10: ROC curves for the MLP and logistic regression with Youden's J operating point",cap),
 P(f"<b>Performance Interpretation:</b> The MLP achieves a test ROC-AUC of <b>{m['auc']:.4f}</b>, clearly above random guessing (0.5000), and the logistic regression baseline reaches {l['auc']:.4f}. Youden's J picks a cutoff of {y['thr']}, giving {y['tpr']*100:.1f}% sensitivity at a {y['fpr']*100:.1f}% false-positive rate. This operating point trades recall for far fewer false alarms than the 0.5 cutoff. The two ROC curves overlap closely, indicating the signal is mostly linear."),
 P("8.5 Comprehensive Out-of-Sample Benchmark Comparison Table",h2),P(f"Table 8: Test-set benchmark – MLP vs Logistic Regression baseline (n = {sp['test'][0]}, 0.5 threshold)",tcap)]
def row(n,k,pct=False):
    a,b_=m[k],l[k]; f=(lambda v:f"{v*100:.2f}%") if pct else (lambda v:f"{v:.4f}")
    d=(a-b_)*100 if pct else (a-b_); return [n,f(a),f(b_),(f"{d:+.2f} pts" if pct else f"{d:+.4f}")]
rows=[["Evaluation Metric","MLP","Logistic Regression","Delta (MLP − LR)"],row("Overall Accuracy","acc",1),row("Precision (Positive)","prec",1),row("Recall / Sensitivity","rec",1),row("Specificity","spec",1),row("F1-Score","f1"),row("ROC-AUC","auc"),row("Average Precision (PR-AUC)","ap"),["Majority-class accuracy",f"{maj:.2f}%",f"{maj:.2f}%","—"]]
st+=[tbl(rows,[5*cm,3.5*cm,3.5*cm,W-12*cm]),Spacer(1,.2*cm),P("Comparative Synthesis",h2),
 P(f"The neural network does <b>not</b> outperform the linear baseline on this dataset: logistic regression is marginally ahead on ROC-AUC ({l['auc']:.4f} vs {m['auc']:.4f}), F1 and accuracy, while the two tie on specificity. With 300 test rows, differences of this size are within sampling noise, so the fair reading is that both models perform similarly. Both beat the majority-class baseline in AUC, but accuracy gains are small ({(m['acc']*100-maj):+.1f} points over {maj:.1f}%). This is consistent with how the data were generated: repeat purchase depends mostly on additive effects (loyalty, shipping speed, Prime membership) plus substantial random noise, a setting where extra non-linear capacity adds little.")]
# ---------- 9 Conclusion ----------
st+=[PageBreak(),P("9. Conclusion &amp; Future Scope",h1),P("9.1 Key Empirical Findings &amp; Patterns",h2),
 P(f"<b>1. Loyalty and Prime membership matter most.</b> Prime customers repeat at {R['rate_prime']['1']}% versus {R['rate_prime']['0']}% (χ² = {ch['is_prime']['chi2']}, p {pv(ch['is_prime']['p'])}), and repeat customers have placed more prior orders (mean {tv['prior_orders']['m1']:.2f} vs {tv['prior_orders']['m0']:.2f}; p {pv(tv['prior_orders']['p'])})."),
 P(f"<b>2. Delivery speed protects retention.</b> Repeat customers waited {tv['shipping_days']['m1']:.2f} days on average versus {tv['shipping_days']['m0']:.2f}, and slower delivery lowers ratings (r = {R['ship_rating_r'][0]})."),
 P(f"<b>3. Revenue is concentrated.</b> Electronics accounts for {ec/rt*100:.1f}% of revenue but has the lowest repeat rate ({R['rate_by_cat']['Electronics']}%); Beauty and Books retain best. November and December together bring {nd/rt*100:.1f}% of revenue."),
 P("9.2 Model Performance Summary",h2),
 P(f"The tuned MLP ({bk.split(' (')[0]}) reached {m['acc']*100:.2f}% test accuracy, {m['rec']*100:.2f}% recall, {m['spec']*100:.2f}% specificity and ROC-AUC {m['auc']:.4f} on unseen data, comparable to logistic regression ({l['auc']:.4f}). Imputation and scaling were fitted on the training partition only, and the test set was used once, for final evaluation."),
 P("9.3 Technical Challenges Faced",h2),
 P("• Heavy right-skew in order value and unit price required log-scaled plots and median-based summaries."),
 P("• Moderate class imbalance means accuracy must be compared with the majority-class baseline."),
 P("• A small dataset (2,000 orders) with noisy labels limits what any model can learn; the MLP over-fit after about nine epochs."),
 P("• TensorFlow/Keras was unavailable, so scikit-learn's MLP was used and L2 regularization replaced dropout."),
 P("9.4 Future Improvements &amp; Technical Extensions",h2),
 P("• <b>Real-data validation:</b> repeat the pipeline on a public retail dataset (for example Olist or Superstore) to confirm the patterns."),
 P("• <b>Gradient boosting and calibration:</b> compare against tree ensembles and calibrate probabilities before using them for targeting."),
 P("• <b>Cost-sensitive thresholding:</b> choose the cutoff from the expected value of a retention offer instead of defaulting to 0.5."),
 P("• <b>Interactive dashboard:</b> publish the figures as a filterable dashboard to complete Fry's Interact stage.")]
# ---------- 10 Refs ----------
st+=[PageBreak(),P("10. Literature Review &amp; References",h1),P("The methodology, statistical framework and visualization workflow follow the established literature below:"),
 P("[1] B. Fry, <i>Visualizing Data: Exploring and Explaining Data with the Processing Environment</i>. Sebastopol, CA: O'Reilly Media, 2008."),
 P("[2] J. W. Tukey, <i>Exploratory Data Analysis</i>. Reading, MA: Addison-Wesley, 1977."),
 P("[3] B. L. Welch, \"The generalization of 'Student's' problem when several different population variances are involved,\" <i>Biometrika</i>, vol. 34, no. 1–2, pp. 28–35, 1947."),
 P("[4] H. Cramér, <i>Mathematical Methods of Statistics</i>. Princeton, NJ: Princeton University Press, 1946."),
 P("[5] I. Goodfellow, Y. Bengio, and A. Courville, <i>Deep Learning</i>. Cambridge, MA: MIT Press, 2016."),
 P("[6] F. Pedregosa, G. Varoquaux, A. Gramfort, et al., \"Scikit-learn: Machine Learning in Python,\" <i>Journal of Machine Learning Research</i>, vol. 12, pp. 2825–2830, 2011."),
 P("[7] W. McKinney, \"Data structures for statistical computing in Python,\" in <i>Proc. 9th Python in Science Conference</i>, 2010, pp. 56–61."),
 P("[8] J. D. Hunter, \"Matplotlib: A 2D graphics environment,\" <i>Computing in Science &amp; Engineering</i>, vol. 9, no. 3, pp. 90–95, 2007.")]
# ---------- 11 Appendix ----------
code=open("analysis.py").read().replace("\t","    ")
lines=[]
for ln in code.splitlines():
    if len(ln)<=118: lines.append(ln)
    else:
        ind=len(ln)-len(ln.lstrip()); w=textwrap.wrap(ln,118,subsequent_indent=" "*(ind+4),break_long_words=True)
        lines.extend(w)
cs=ParagraphStyle("code",fontName=FONT_MONO,fontSize=5.6,leading=6.6)
st+=[PageBreak(),P("11. Appendix: Core Reproducible Code Architecture",h1),
 P("The full pipeline, covering data generation, preprocessing, statistical tests, all figures, MLP tuning, evaluation and permutation importance, is one script (analysis.py, run with Python 3.12, scikit-learn, pandas, SciPy, seaborn and matplotlib). It writes the dataset CSV, ten figures and results.json. Long lines are wrapped for print.")]
chunk=88
for i in range(0,len(lines),chunk):
    if i>0: st+=[PageBreak(),P("11. Appendix: Core Code Architecture (Cont.)",h1)]
    st.append(Preformatted("\n".join(lines[i:i+chunk]),cs))
st+=[Spacer(1,.4*cm),P(f"<b>Academic Declaration:</b> This case study report, exploratory visual artifacts and neural network implementation were developed by {NAME} ({ROLL}) for course U21ADP05 at KPR Institute of Engineering and Technology. The dataset is synthetic and all code and figures are reproducible by running the script above.",body)]
doc.build(st); print("built")
