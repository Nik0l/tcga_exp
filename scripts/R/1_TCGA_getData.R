# Scripts to get TCGA molecular data (RNAseq and mutations)

library(readr)

setwd("/Users/sophie/Dropbox (The Francis Crick)/Sophie/Astra_project/Repositories/tcga_exp")
output_path <- "data/sophie/"

# Function definitions #########################################################  

getGeneExp <- function(project, download = FALSE, destination_folder) {
  # Queries GDC database and returns dataframe with list of RNAseq raw (non-normalized, HTseq counts) counts files available
  # Also has the option to download the data - in that case, destination_folder must be provided
  library(TCGAbiolinks)
  
  query <- GDCquery(project = project,
                    data.category = "Transcriptome Profiling",
                    data.type = "Gene Expression Quantification",
                    workflow.type = "HTSeq - Counts",
                    legacy = FALSE
  )
  
  exp_df <- getResults(query)
  
  if (download) {
    setwd(destination_folder)
    GDCdownload(query_exp, method = "api")
  }
  
  return(exp_df)
}


getMut <- function(project) {
  # Gets MAF file from MC3 Pan-Cancer study (https://gdc.cancer.gov/about-data/publications/mc3-2017)
  # Returns a dataframe with an added column (cases.submitter_id) corresponding to patient ID
  library(TCGAmutations)
  library(dplyr)
  
  maf <- tcga_load(study = project)
  maf_df <- maf@data %>% 
    mutate(cases.submitter_id = substr(Tumor_Sample_Barcode, 1, 12))
  
  return(maf_df)
}


# Get and save RNAseq dataframes ###############################################

lung_rna_gdc <- getGeneExp(project = c("TCGA-LUAD", "TCGA-LUSC"))
lusc_rna_gdc <- getGeneExp(project = "TCGA-LUSC")
luad_rna_gdc <- getGeneExp(project = "TCGA-LUAD")

write_csv(lung_rna_gdc, file = paste(output_path, "all_lung_rna_gdc.csv", sep=""))
write_csv(lusc_rna_gdc, file = paste(output_path, "lusc_rna_gdc.csv", sep=""))
write_csv(luad_rna_gdc, file = paste(output_path, "luad_rna_gdc.csv", sep=""))


# Get and save mutation data ##################################################
lusc_mut <- getMut("LUSC")
luad_mut <- getMut("LUAD")

write_csv(lusc_mut, file = paste(output_path, "lusc_mut_mc3.csv", sep=""))
write_csv(luad_mut, file = paste(output_path, "luad_mut_mc3.csv", sep=""))


