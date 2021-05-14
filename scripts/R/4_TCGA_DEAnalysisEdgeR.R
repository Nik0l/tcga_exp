# Performs edgeR DE analysis

setwd("/Users/sophie/Dropbox (The Francis Crick)/Sophie/Astra_project/Repositories/tcga_exp")

library(readr)
library(dplyr)
library(edgeR)
library(gridExtra)

# Function definitions ########################################################

prepareDGEList <- function(counts_table, sample_info, norm_method="TMM", tumor_only=FALSE){
  #' prepares DGEList object for projection and DE analysis
  #' performs count normalization
  #' @param counts_table a dataframe with RNAseq raw counts, with sample barcodes as columns. Columns 1 and 2 correspond to 
  #' gene_symbols and ensembl_gene_ids respectively
  #' @param sample_info a dataframe with the names of counts files to be considered
  #' @param norm_method a string specifying the normalization method to be used (for CalcNormFactors {edgeR} function)
  #' @param tumor_only specifies whether to discard normal tissue samples
  
  if(tumor_only){
    
    # remove non-tumor samples from counts_table
    normal_samples <- sample_info %>% 
      filter(tumor == 0) %>% 
      pull(cases)
    
    to_delete <- which(colnames(counts_table) %in% normal_samples)
    counts_table <- counts_table[,-to_delete]
    
    # filter sample_info to include only tumor samples
    sample_info <- sample_info %>% 
      filter(tumor == 1)
  }
  
  groups <- sample_info$mutation_status
  
  # create DGEList object
  y <- DGEList(counts = counts_table[,3:ncol(counts_table)], genes = counts_table[,2], group = groups)
  y$genes$Symbol <- counts_table$external_gene_name
  
  # keep gene with cpm>10 in at least 70% of samples (per group)
  keep <- filterByExpr(y)
  y <- y[keep, , keep.lib.sizes=FALSE]
  dim(y)
  
  # Remove duplicated genes by keeping the entries with the highest counts
  o <- order(rowSums(y$counts), decreasing=TRUE)
  y <- y[o,]
  d <- duplicated(y$genes$Symbol)
  y <- y[!d,]
  
  # Re-compute library sizes after the filtering
  y$samples$lib.size <- colSums(y$counts)
  
  # Rename y$counts and y$genes rows as Ensembl IDs and remove the genes$genes column
  rownames(y$counts) <- rownames(y$genes) <- y$genes$genes
  y$genes$genes <- NULL
  
  y <- calcNormFactors(y, method = norm_method)
  
  return(y)
  
}


DEAnalysis <- function(y){
  #' performs edgeR DE analysis using the exact test method (comparison between 2 groups)
  #' @param y a DGEList object containing the group variable
  #' @param n_top_genes number of DE genes to be output in a table format
  #' @param by parameter used to select DE genes
  #' @param cutoff threshold to select DE genes
  
  # creates design matrix
  design <- model.matrix(~0+group, data=y$samples)
  colnames(design) <- c("WT","MUT")
  
  # estimates dispersion (technical and biological variability)
  print("Estimating dispersion...")
  y <- estimateDisp(y, design)
  
  # performs exact test
  print("Performing exact test...")
  et <- exactTest(y, pair=c(1,2))
  assign("et",et,.GlobalEnv)
  
  # generates table containing all genes and their respective test values
  all_genes <- topTags(et, n = dim(y)[1])$table
  
  return(all_genes)
}


# load data ####################################################################
counts_table_pc <- read_csv("data/sophie/luad_counts_table_pc.csv")
sample_info <- read_csv("data/sophie/luad_rna_sample_info.csv")

# run DE analysis #############################################################
y <- prepareDGEList(counts_table_pc, sample_info, tumor_only=FALSE)
all_genes_et <- DEAnalysis(y)
write_csv(all_genes_et, "results/edgeR/ET_pc_genes.csv")

yy <- prepareDGEList(counts_table_pc, sample_info, tumor_only=TRUE)
all_genes_et_2 <- DEAnalysis(yy)
write_csv(all_genes_et_2, "results/edgeR/ET_pc_genes_tumor_only.csv")

# Explore results #############################################################
# make table with top 20 genes -- all samples
de_genes <- all_genes_et %>% 
  filter(FDR <= 0.01) %>% 
  arrange("FDR")

de_genes[,3:4] <- round(de_genes[,3:4], digits = 3)
de_genes[,5:6] <- signif(de_genes[,5:6], digits=3)

dev.off()
grid.table(de_genes[1:20,], rows=NULL)

# compare with kaufman signature
kaufman_genes <- c('AVPI1', 'BAG1', 'CPS1', 'DUSP4', 'FGA', 'GLCE', 'HAL', 'IRS2', 
                   'MUC5AC', 'PDE4D', 'PTP4A1', 'RFK', 'SIK1', 'TACC2', 'TESC', 'TFF1')

k_genes <- all_genes_et %>% 
  filter(Symbol %in% kaufman_genes)

k_genes[,3:4] <- round(k_genes[,3:4], digits = 3)
k_genes[,5:6] <- signif(k_genes[,5:6], digits=3)

dev.off()
grid.table(k_genes, rows=NULL)



# make table with top 20 genes -- tumor only
de_genes <- all_genes_et_2 %>% 
  filter(FDR <= 0.01) %>% 
  arrange("FDR")

de_genes[,3:4] <- round(de_genes[,3:4], digits = 3)
de_genes[,5:6] <- signif(de_genes[,5:6], digits=3)

dev.off()
grid.table(de_genes[1:20,], rows=NULL)

# compare with kaufman signature
kaufman_genes <- c('AVPI1', 'BAG1', 'CPS1', 'DUSP4', 'FGA', 'GLCE', 'HAL', 'IRS2', 
                   'MUC5AC', 'PDE4D', 'PTP4A1', 'RFK', 'SIK1', 'TACC2', 'TESC', 'TFF1')

k_genes <- all_genes_et %>% 
  filter(Symbol %in% kaufman_genes)

k_genes[,3:4] <- round(k_genes[,3:4], digits = 3)
k_genes[,5:6] <- signif(k_genes[,5:6], digits=3)

dev.off()
grid.table(k_genes, rows=NULL)



