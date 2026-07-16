# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:45:28 2026

@author: ebebo
"""

# IMPORT & CONFIGURATION

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler

# set the plotly theme as default
pio.templates.default = 'plotly_white'

# configuration of the page
st.set_page_config(
    page_title="F1 Telemetry Dashboard",
    page_icon="🏎️",
    layout="wide"
)


# DOWNLOADING & CLEANING
@st.cache_data  # store the data in cache

def load_data():
    
    df = pd.read_csv('All Laps - 5 June 2026.csv')
    
    df['interval_data'] = df['interval_data'].apply(
        lambda x : json.loads(x) if pd.notnull(x) else [])
    
    df_exp = df.explode('interval_data').reset_index(drop=True)
    telem = pd.json_normalize(df_exp['interval_data'])
    final = pd.concat([df_exp.drop(columns=['interval_data']), telem], axis = 1)
    
    # save metadata
    season = final['season'].iloc[0]
    event = final['event_name'].iloc[0]
    circuit = final['circuit'].iloc[0]
    
    const_cols = [c for c in final.columns if final[c].nunique() == 1]
    final = final.drop(columns=const_cols)
    
    final['throttle_open'] = final['throttle'] > 0
    final['is_braking'] = final['brake'] == 1
    final['brake'] = final['brake'].astype(int)
    
    best = final.groupby('driver_code')['lap_time_sec_total'].transform('min')
    final['is_best_lap'] = final['lap_time_sec_total'] == best
    
    final['speed_range'] = pd.cut(
        final['speed'], 
        bins=[0, 100, 200, 300, 400], 
        labels=['0-100', '100-200', '200-300', '300+']
    )
    final['speed_range'] = final['speed_range'].astype(str)
    
    return final, season, event, circuit


# upload data
final_df, SEASON, EVENT, CIRCUIT = load_data()


# TITLE & SIDEBAR
# principal title
st.title("🏎️ F1 Telemetry Analysis")
st.markdown(f"**{EVENT} {SEASON} — {CIRCUIT}** | Pre-Season Testing")
st.divider()

# side bar with filters
with st.sidebar:
    st.header("🔧 Filters")
    
    all_drivers = sorted(final_df['driver_code'].unique().tolist())
    
    selected_drivers = st.multiselect(
        "Seleziona piloti:",
        options=all_drivers,
        default=all_drivers[:5]
    )
    
    st.divider()

    selected_single = st.selectbox(
    "🔍 Select a single driver (for Q6 Speed Distribution & Q10 Driving Style):",
    options=all_drivers,
    index=0
    )
    
    st.divider()
    st.caption("F1 Telemetry Dashboard — TU Berlin 2026")
    
# filter data based on the selection
if selected_drivers:
    df = final_df[final_df['driver_code'].isin(selected_drivers)]
else:
    df = final_df.copy()
    
    
    
# UPLOADING ALL THE DATAFRAMES
# ── DATAFRAMES ────────────────────────────────────────────────

# Q1 — Best lap time per driver
best_laps = (df.groupby('driver_code')['lap_time_sec_total']
               .min()
               .reset_index()
               .rename(columns={'lap_time_sec_total': 'best_lap_time'})
               .sort_values('best_lap_time')
               .reset_index(drop=True))

# Q2
lap_evo_stats = (df[df['lap_time_sec_total'] < 200]
                 .groupby(['driver_code', 'lap_number'])['lap_time_sec_total']
                 .agg(mean_time='mean', std_time='std', n='count')
                 .reset_index())
lap_evo_stats['ci'] = 1.96 * (lap_evo_stats['std_time'] / np.sqrt(lap_evo_stats['n']))

# Q3 — Driver consistency (solo giri < 200s)
consistency = (df[df['lap_time_sec_total'] < 200]
               .groupby('driver_code')['lap_time_sec_total']
               .agg(best_lap='min', worst_lap='max')
               .reset_index())
consistency['delta'] = (consistency['worst_lap'] - consistency['best_lap']).round(2)
consistency = consistency.sort_values('delta').reset_index(drop=True)
consistency['se'] = consistency['delta'] / np.sqrt(
    df.groupby('driver_code')['lap_number'].nunique().reset_index()['lap_number'])
consistency['ci'] = 1.96 * consistency['se']

# Q4 — Bubble chart (speed_analysis per il merge)
speed_analysis = (df.groupby('driver_code')['speed']
                    .agg(max_speed='max', min_speed='min', avg_speed='mean')
                    .reset_index()
                    .sort_values('max_speed', ascending=False))
speed_analysis['avg_speed'] = speed_analysis['avg_speed'].round(2)
bubble_df = (best_laps
             .merge(speed_analysis[['driver_code', 'max_speed']], on='driver_code')
             .merge(consistency[['driver_code', 'delta']], on='driver_code'))

# Q5 — Speed analysis full (gear > 0)
speed_analysis_full = (df[df['gear'] > 0]
                         .groupby('driver_code')['speed']
                         .agg(max_speed='max', avg_speed='mean')
                         .reset_index())
speed_analysis_full['avg_speed'] = speed_analysis_full['avg_speed'].round(2)
speed_long = pd.melt(
    speed_analysis_full[['driver_code', 'max_speed', 'avg_speed']],
    id_vars='driver_code',
    var_name='speed_type',
    value_name='speed'
)

# Q6 — Speed distribution
speed_dist = (df.groupby(['driver_code', 'speed_range'], observed=True)
                .size()
                .reset_index()
                .rename(columns={0: 'seconds'}))
speed_dist['speed_range'] = speed_dist['speed_range'].astype(str)

# Q7 — Speed per gear
speed_per_gear = (df[df['gear'] > 0]
                    .groupby('gear')['speed']
                    .mean()
                    .reset_index()
                    .rename(columns={'speed': 'avg_speed'}))
speed_per_gear['avg_speed'] = speed_per_gear['avg_speed'].round(2)

# Q8 — Throttle percentage
perc_throttle_open = (df.groupby('driver_code')['throttle_open']
                        .mean() * 100).round(2).reset_index()
perc_throttle_open.rename(columns={'throttle_open': 'throttle_pct'}, inplace=True)
perc_throttle_open.sort_values('throttle_pct', ascending=False, inplace=True)
perc_throttle_open.reset_index(drop=True, inplace=True)

# Q9 — Brake percentage
perc_brake_open = (df.groupby('driver_code')['is_braking']
                     .mean() * 100).round(2).reset_index()
perc_brake_open.rename(columns={'is_braking': 'braking_pct'}, inplace=True)
perc_brake_open.sort_values('braking_pct', ascending=False, inplace=True)
perc_brake_open.reset_index(drop=True, inplace=True)

# Q10 — Driving style (best lap)
driving_style = (df.loc[df['is_best_lap'] == True]
                   [['driver_code', 't', 'throttle', 'brake', 'speed']]
                   .reset_index(drop=True))
driving_style_long = pd.melt(
    driving_style,
    id_vars=['driver_code', 't'],
    value_vars=['throttle', 'brake'],
    var_name='metric',
    value_name='value'
)

# Q11 — RPM vs Throttle
rpm_vs_throttle = (df[df['gear'] > 0]
                     [['driver_code', 'rpm', 'throttle']]
                     .reset_index(drop=True))

# Q12 — Average RPM
avg_rpm = (df[df['gear'] > 0]
             .groupby('driver_code')['rpm']
             .mean()).round(2).reset_index()
avg_rpm.rename(columns={'rpm': 'avg_rpm'}, inplace=True)
avg_rpm.sort_values('avg_rpm', ascending=False, inplace=True)
avg_rpm.reset_index(drop=True, inplace=True)

# Q13 — RPM per gear
rpm_distribution_perGear = (df[df['gear'] > 0]
                               [['gear', 'rpm']]
                               .reset_index(drop=True))

# Q14 — Correlation matrix
num_cols = ['speed', 'rpm', 'throttle', 'brake', 'gear', 'lap_time_sec_total']
corr_matrix = df[num_cols].corr().round(2)

# Q15 — Telemetry profile
telemetry_profile = (df[df['is_best_lap'] == True]
                       [['driver_code', 't', 'speed', 'rpm', 'throttle']]
                       .reset_index(drop=True))
scaler = MinMaxScaler()
telemetry_profile[['speed_norm', 'rpm_norm', 'throttle_norm']] = scaler.fit_transform(
    telemetry_profile[['speed', 'rpm', 'throttle']]
)
telemetry_long = pd.melt(
    telemetry_profile,
    id_vars=['driver_code', 't'],
    value_vars=['speed_norm', 'rpm_norm', 'throttle_norm'],
    var_name='metric',
    value_name='value'
)

# VISUALIZATION OF THE DATAFRAMES
# ── SEZIONE 1 — LAP TIMES & PERFORMANCE ──────────────────────
st.header("🏁 Group 1 — Lap Times & Performance")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.bar(
        data_frame=best_laps,
        x='best_lap_time', y='driver_code', orientation='h',
        title='(1) Driver Lap Time Comparison',
        labels={'best_lap_time': 'Best Lap Time (s)', 'driver_code': ''},
        color='best_lap_time',
        color_continuous_scale=px.colors.sequential.Blues_r)
    fig1.update_layout(title_font=dict(size=16, weight='bold'),
                       coloraxis_showscale=False,
                       yaxis=dict(categoryorder='total ascending'))
    st.plotly_chart(fig1, use_container_width=True)
    with st.expander("Interpretation"):
        st.write("""
        This chart shows the **best lap time** recorded by each driver.
        - 🥇 The driver at the top recorded the fastest single lap
        - Only the best lap per driver is shown — not the average
        - Darker color = faster lap time
        """)

with col2:
    fig3 = px.bar(
        data_frame=consistency,
        x='driver_code', y='delta', error_y='ci',
        title='(3) Driver Consistency',
        labels={'driver_code': '', 'delta': 'Lap Time Delta (s)'},
        color='delta',
        color_continuous_scale=px.colors.sequential.Reds)
    fig3.update_layout(title_font=dict(size=16, weight='bold'),
                       coloraxis_showscale=False,
                       xaxis=dict(categoryorder='total ascending'))
    st.plotly_chart(fig3, use_container_width=True)
    with st.expander("Interpretation"):
        st.write("""
        Delta between each driver's fastest and slowest valid lap (under 200s).
        - Small delta = consistent driver ✅
        - Large delta = inconsistent driver ❌
        - Error bars = 95% Confidence Interval
        """)

fig2 = px.bar(
    data_frame=lap_evo_stats,
    x='lap_number', y='mean_time',
    error_y='ci',
    color='driver_code',
    barmode='group',
    title='(2) Lap Time Evolution',
    labels={'mean_time': 'Mean Lap Time (s)',
            'lap_number': 'Lap Number',
            'driver_code': 'Driver'})
fig2.update_layout(
    title_font=dict(size=16, weight='bold'),
    xaxis=dict(tickmode='linear', dtick=1))
st.plotly_chart(fig2, use_container_width=True)
with st.expander("Interpretation"):
    st.write("""
    Mean lap time per lap number for each driver with 95% CI.
    - Each bar = mean lap time for that lap
    - Error bars = 95% Confidence Interval
    - Bars going down = driver improving lap after lap
    - Large CI = high variability in that lap across drivers
    """)

fig4 = px.scatter(
    data_frame=bubble_df,
    x='best_lap_time', y='max_speed', size='delta',
    color='driver_code', hover_name='driver_code',
    title='(4) Driver Performance Overview',
    labels={'best_lap_time': 'Best Lap Time (s)',
            'max_speed': 'Top Speed (km/h)',
            'delta': 'Consistency (s)', 'driver_code': 'Driver'},
    size_max=40)
fig4.update_layout(title_font=dict(size=16, weight='bold'))
st.plotly_chart(fig4, use_container_width=True)
with st.expander("Interpretation"):
    st.write("""
    Three performance dimensions in one view. Each bubble = one driver.
    - X axis → Best lap time (further left = faster)
    - Y axis → Top speed (higher = faster)
    - Bubble size → Consistency delta (bigger = less consistent)
    - 🏆 Ideal driver: bottom-left with a small bubble
    """)

st.divider()

# ── SEZIONE 2 — SPEED ANALYSIS ────────────────────────────────
st.header("⚡ Group 2 — Speed Analysis")

fig5 = px.bar(
    data_frame=speed_long,
    x='driver_code', y='speed', color='speed_type', barmode='group',
    title='(5) Top and Average Speed Per Driver',
    labels={'driver_code': '', 'speed': 'Speed (km/h)', 'speed_type': 'Speed Type'},
    color_discrete_map={'max_speed': '#1f77b4', 'avg_speed': '#ff7f0e'})
fig5.update_layout(title_font=dict(size=16, weight='bold'),
                   xaxis=dict(categoryorder='total descending'))
st.plotly_chart(fig5, use_container_width=True)
with st.expander("Interpretation"):
    st.write("""
    Two speed metrics compared for each driver:
    - 🔵 Max speed — highest speed recorded during the session
    - 🟠 Avg speed — average speed while in gear (gear > 0)
    - Large gap between the two = fast on straights but slower through corners
    """)

col1, col2 = st.columns(2)

with col1:
    fig7 = px.bar(
        data_frame=speed_per_gear,
        x='gear', y='avg_speed',
        title='(7) Average Speed Per Gear',
        labels={'gear': 'Gear', 'avg_speed': 'Average Speed (km/h)'},
        color='avg_speed',
        color_continuous_scale=px.colors.sequential.Blues)
    fig7.update_layout(title_font=dict(size=16, weight='bold'),
                       coloraxis_showscale=False,
                       xaxis=dict(tickmode='linear', dtick=1))
    st.plotly_chart(fig7, use_container_width=True)
    with st.expander("Interpretation"):
        st.write("""
        Average speed reached in each gear across all drivers and laps.
        - Higher gears = higher speeds (as expected)
        - Gear 1 = slow corners and standing starts
        - Gear 8 = top speed on long straights
        - Gear 0 (stationary) excluded from analysis
        """)

with col2:
    # Q6 — filtra per il singolo pilota
    speed_dist_single = speed_dist[speed_dist['driver_code'] == selected_single]

    fig6 = px.sunburst(
        data_frame=speed_dist_single,
        path=['driver_code', 'speed_range'],
        values='seconds', color='speed_range',
        color_discrete_map={'0-100': '#636EFA', '100-200': '#EF553B',
                        '200-300': '#00CC96', '300+': '#FFA15A'},
        title=f'(6) Speed Distribution — {selected_single}')
    fig6.update_layout(title_font=dict(size=16, weight='bold'))
    fig6.update_traces(hovertemplate=(
        "<b>%{label}</b><br>Seconds: %{value}<br>"
        "Share of total: %{percentRoot:.1%}<extra></extra>"))
    st.plotly_chart(fig6, use_container_width=True)

st.divider()

# ── SEZIONE 3 — THROTTLE & BRAKE ──────────────────────────────
st.header("🔥 Group 3 — Throttle & Brake")

col1, col2 = st.columns(2)

with col1:
    fig8 = px.bar(
        data_frame=perc_throttle_open,
        x='throttle_pct', y='driver_code', orientation='h',
        title='(8) Throttle Open Percentage Per Driver',
        labels={'throttle_pct': 'Throttle Open (%)', 'driver_code': ''},
        color='throttle_pct',
        color_continuous_scale=px.colors.sequential.Greens)
    fig8.update_layout(title_font=dict(size=16, weight='bold'),
                       coloraxis_showscale=False,
                       yaxis=dict(categoryorder='total ascending'))
    st.plotly_chart(fig8, use_container_width=True)
    with st.expander("Interpretation"):
        st.write("""
        Percentage of time each driver had the throttle engaged (throttle > 0).
        - Higher % = more time accelerating ✅
        - In F1, drivers are typically on throttle 60-75% of the lap
        - Differences reflect driving style or the programme run
        """)

with col2:
    fig9 = px.bar(
        data_frame=perc_brake_open,
        x='braking_pct', y='driver_code', orientation='h',
        title='(9) Braking Percentage Per Driver',
        labels={'braking_pct': 'Braking Time (%)', 'driver_code': ''},
        color='braking_pct',
        color_continuous_scale=px.colors.sequential.Reds)
    fig9.update_layout(title_font=dict(size=16, weight='bold'),
                       coloraxis_showscale=False,
                       yaxis=dict(categoryorder='total ascending'))
    st.plotly_chart(fig9, use_container_width=True)
    with st.expander("Interpretation"):
        st.write("""
        Percentage of time each driver was actively braking.
        - Compare with throttle chart (Q8):
        - High throttle + Low brake = smooth driving style 🟢
        - High throttle + High brake = aggressive in both directions 🟡
        - Low throttle + High brake = cautious or many slow laps 🔴
        """)

# Q10 — filtra per il singolo pilota
driving_style_single = driving_style_long[
    driving_style_long['driver_code'] == selected_single]

fig10 = px.line(
    data_frame=driving_style_single,
    x='t', y='value', color='metric',
    title=f'(10) Driving Style — {selected_single} Fastest Lap',
    labels={'t': 'Time (s)', 'value': 'Value', 'metric': 'Metric'})
fig10.update_layout(title_font=dict(size=16, weight='bold'))
st.plotly_chart(fig10, use_container_width=True)
with st.expander("📖 How to read this chart"):
    st.write("""
             Second-by-second throttle and brake profile on the selected driver's fastest lap.
             - 🔵 Throttle — gas pedal input (0-100%)
             - 🔴 Brake — braking input (0 = no brake, 1 = braking)
             - Use the sidebar filter to switch between drivers
             """)

st.divider()

# ── SEZIONE 4 — RPM & GEAR ────────────────────────────────────
st.header("⚙️ Group 4 — RPM & Gear Telemetry")

col1, col2 = st.columns(2)

with col1:
    fig12 = px.bar(
        data_frame=avg_rpm,
        x='avg_rpm', y='driver_code', orientation='h',
        title='(11) Average RPM Per Driver',
        labels={'avg_rpm': 'Average RPM', 'driver_code': ''},
        color='avg_rpm',
        color_continuous_scale=px.colors.sequential.Oranges)
    fig12.update_layout(title_font=dict(size=16, weight='bold'),
                        coloraxis_showscale=False,
                        yaxis=dict(categoryorder='total ascending'))
    st.plotly_chart(fig12, use_container_width=True)
    with st.expander("Interpretation"):
        st.write("""
        Average RPM maintained by each driver (gear > 0 only).
        - Higher RPM = more aggressive driving style or
          more time on high-speed sections
        - Lower RPM = more conservative driving or
          more time on slow/installation laps
        """)

with col2:
    fig13 = px.box(
        data_frame=rpm_distribution_perGear,
        x='gear', y='rpm', color='gear',
        title='(12) RPM Distribution Per Gear',
        labels={'gear': 'Gear', 'rpm': 'RPM'})
    fig13.update_layout(title_font=dict(size=16, weight='bold'),
                        showlegend=False,
                        xaxis=dict(tickmode='linear', dtick=1))
    st.plotly_chart(fig13, use_container_width=True)
    with st.expander("Interpretation"):
        st.write("""
        Distribution of RPM values for each gear across all drivers and laps.
        - Box = interquartile range (25th to 75th percentile)
        - Line inside box = median RPM for that gear
        - Whiskers = min and max values
        - Dots outside whiskers = outliers
        - Overlap between gears = typical RPM range for gear changes
        """)


st.divider()

# ── SEZIONE 5 — CROSS-VARIABLE ────────────────────────────────
st.header("🔬 Group 5 — Single-Lap Telemetry Pro")

col1, col2 = st.columns([1, 2])


fig15 = px.line(
    data_frame=telemetry_long,
    x='t', y='value', color='metric',
    facet_col='driver_code', facet_col_wrap=3,
    title='(13) Single-Lap Telemetry Profile — Fastest Lap Per Driver',
    labels={'t': 'Time (s)', 'value': 'Normalised Value (0-1)', 'metric': 'Metric'})
fig15.update_layout(title_font=dict(size=16, weight='bold'))
fig15.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
st.plotly_chart(fig15, use_container_width=True)
with st.expander("Interpretation"):
    st.write("""
    Second-by-second telemetry on each driver's fastest lap.
    All metrics normalised 0-1 to be comparable on the same scale.
    - 🔵 speed_norm → normalised speed
    - 🟠 rpm_norm → normalised RPM
    - 🟢 throttle_norm → normalised throttle

    What to look for:
    - Speed drops → corner entry (braking zone)
    - Throttle rises → corner exit (acceleration)
    - Differences between drivers = different driving styles
    """)

st.divider()
st.caption("Francesco Bovina — Data Analysis and Visualization with Python — TU Berlin 2026")
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

