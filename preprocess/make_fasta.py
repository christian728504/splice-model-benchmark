import subprocess
import os


def make_fasta(output_dir = "reference_files"):
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    download_variants = "wget https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/NA12878_HG001/latest/GRCh38/HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
    download_reference_genome = "wget https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/references/GRCh38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta.gz && gunzip GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta"
    get_biallelic_variants = "bcftools view -v snps -m2 -M2 HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz -Oz -o GM12878_SNPs_biallelic.vcf.gz"
    get_index = "bcftools index GM12878_SNPs_biallelic.vcf.gz"
    get_consensus = "bcftools consensus -f GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta -H 1 GM12878_SNPs_biallelic.vcf.gz > GM12878.fasta"

    subprocess.run(download_variants, shell=True)
    subprocess.run(download_reference_genome, shell=True)
    subprocess.run(get_biallelic_variants, shell=True)
    subprocess.run(get_index, shell=True)
    subprocess.run(get_consensus, shell=True)

    os.remove(path="HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz")
    os.remove(path="GM12878_SNPs_biallelic.vcf.gz")
    os.remove(path="GM12878_SNPs_biallelic.vcf.gz.csi")
    
    print(f"Done! Fasta file saved to {output_dir}/GM12878.fasta")