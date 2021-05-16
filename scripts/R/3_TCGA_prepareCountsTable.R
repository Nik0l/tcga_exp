# Prepares counts_table

setwd("/Users/sophie/Dropbox (The Francis Crick)/Sophie/Astra_project/Repositories/tcga_exp")

library(readr)
library(biomaRt)


# Function definitions ########################################################

sampleInfo <- function(patient_list, rna_df){
  #' creates dataframe with sample barcodes and corresponding mutation status and sample type (tumor vs normal)
  #' also reorders table to place WT samples first
  #' @param patient_list a dataframe containing patient IDs "cases.submitter_id" and mutation_status
  #' @param rna_df a dataframe from GDC containing patient IDs "cases.submitter_id", sample_type and sample barcodes "cases"
  
  df <- full_join(rna_df[, c("cases.submitter_id", "sample_type", "cases", "id")], 
                  patient_list, by = "cases.submitter_id") %>% 
    filter(cases.submitter_id %in% patient_list$cases.submitter_id)
  
  df <- df %>% 
    mutate(tumor = case_when(sample_type == 'Solid Tissue Normal' ~ 0, TRUE ~ 1)) %>% 
    mutate(mutation_status = case_when(sample_type == 'Solid Tissue Normal' ~ 0, TRUE ~ mutation_status)) %>% 
    relocate(cases) %>% 
    dplyr::select(-sample_type) %>% 
    arrange(mutation_status)
  
  return(df)
}


generateCountsTable <- function(sample_info, rna_df, GDCdata_path){
  
  #' Prepares a counts table for DE analysis
  #' @param sample_info a dataframe with the names of counts files to be considered
  #' @param rna_df a dataframe from GDC containing patient IDs "case.submitter_id" and names of counts files for each patient
  #' @param GDCdata_path path to the GDCdata folder where counts files are stored
  
  # initialize counts_table
  df <- data.frame(ensembl_gene_id = character())
  
  samples <-sample_info$id
  barcodes <- sample_info$cases
  
  # fill counts_table with data from all patients
  for (i in seq_along(samples)) {
    
    file <- list.files(paste(GDCdata_path, "/", samples[i], sep = ""), pattern = "*.gz")
    barcode <- barcodes[i]
    
    data <- read.delim(paste(GDCdata_path, "/", samples[i], "/", file, sep = ""), 
                       check.names=FALSE, 
                       stringsAsFactors=FALSE, 
                       header=FALSE, 
                       col.names=c("ensembl_gene_id", barcode))
    
    df <- full_join(df, data, by = "ensembl_gene_id")
    
    print(paste("Fetching RNAseq data... ", i, "/", length(samples)))
    
  }
  
  # counts table cleanup
  # Remove non-gene rows
  rows_to_delete <- which(substring(df$ensembl_gene_id, 1, 1) != "E")
  counts_table <- df[-rows_to_delete,]
  # remove versions from Ensembl ID
  gene_ids <- sub("\\..*", "", counts_table$ensembl_gene_id)
  counts_table$ensembl_gene_id <- gene_ids
  
  return(counts_table)
  
}


filterGenes <- function(counts_table, mart, gene_types){
  #' Adds gene symbols to counts table and selects genes of interest
  #' @param counts_table a dataframe with RNAseq counts for al patients. Needs to contain Ensembl_gene_id column and 
  #' sample barcodes columns (generated with generateCountsTable function)
  #' @param mart reference for gene symbols (generated with BioMart)
  #' @param gene_types an array with the gene_biotypes to be considered (e.g. protein_coding)
  
  # Get info for genes in counts_table, and add it to the table
  genes <- getBM(attributes = c('ensembl_gene_id','external_gene_name', 'gene_biotype', 'description'),
                 filters = 'ensembl_gene_id', 
                 values = counts_table$ensembl_gene_id,
                 mart = mart)
  
  counts_table_genes <- left_join(counts_table, genes[, c('ensembl_gene_id', 'external_gene_name', "gene_biotype")], 
                                  by = 'ensembl_gene_id') %>% 
    filter(gene_biotype %in% gene_types) %>% 
    relocate(external_gene_name) %>% 
    dplyr::select(-gene_biotype)
  
  return(counts_table_genes)
  
}


# Load data ####################################################################
patient_list <- read_csv("data/sophie/luad_patient_list.csv")
rna_df <- read_csv("data/sophie/luad_rna_gdc.csv")
path = "data/sophie/GDCdata/TCGA-LUAD/harmonized/Transcriptome_Profiling/Gene_Expression_Quantification/"
mart <- useMart(biomart = "ensembl", dataset = "hsapiens_gene_ensembl")
gene_types <- c("protein_coding")

# create sample info table and amcounts_table_pc (protein-coding genes only)
sample_info <- sampleInfo(patient_list, rna_df)
write_csv(sample_info, "data/sophie/luad_rna_sample_info.csv")

counts_table <- generateCountsTable(sample_info, rna_df, path)
write_csv(counts_table, "data/sophie/luad_counts_table.csv")

counts_table_pc <- filterGenes(counts_table, mart, gene_types)
write_csv(counts_table_pc, "data/sophie/luad_counts_table_pc.csv")
