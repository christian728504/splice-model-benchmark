import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import io
from PIL import Image

st.set_page_config(page_title="Genomic Volcano Plot Viewer", layout="wide")

# App title and description
st.title("Genomic Volcano Plot Viewer")
st.markdown("Visualize significance vs. effect size for genomic regions across biosampleses")

# Function to load data
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, sep='\t')

# Find all dataset files
# In a real app, replace this with your actual file path pattern
dataset_files = glob("/zata/zippy/ramirezc/splice-model-benchmark/golden_standard/sig_calcs/*.tsv")  # Adjust this pattern to match your files
biosample_term_names = [file_path.split('/')[-1].split('.tsv')[0].replace('_', ' ') for file_path in dataset_files]
biosample_mapping = dict(zip(
    biosample_term_names,
    dataset_files,
))
# Dataset selector
biosample_term = st.selectbox("Select Dataset", sorted(biosample_term_names))

# Load the selected dataset
df = load_data(biosample_mapping.get(biosample_term))

# Add gene and junction columns if they don't exist (for backward compatibility)
if 'gene_name' not in df.columns:
    df['gene_name'] = [f"Gene_{i}" for i in range(len(df))]
    
if 'intron_junction' not in df.columns:
    df['intron_junction'] = df.apply(
        lambda row: f"{row['chrom']}:{row['start']}-{row['end']}({row['strand']})", 
        axis=1
    )

st.subheader("Dataset Information")
st.metric("Significant Regions (adj p<0.05)", len(df[df['adj_pvalue'] < 0.05]))

st.subheader("Data Table")
st.markdown("Filter the data table below:")

selected_gene = st.selectbox(
    "Select a gene to display details:",
    options=['<select>'] + sorted(df['gene_name'].unique().tolist())
)

if selected_gene != '<select>':
    filtered_df = df[
        (df['gene_name'] == selected_gene)
    ]
else:
    filtered_df = df

st.subheader("Filtered Data Table")
st.dataframe(filtered_df[[
    'gene_name', 'intron_junction', 'zscore', 'pvalue', 'adj_pvalue'
]])