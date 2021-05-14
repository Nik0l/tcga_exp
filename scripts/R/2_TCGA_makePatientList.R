# Makes list of patient IDs that have intersectional data and adds theur mutant status

library(readr)
library(dplyr)
library(stringr)

setwd("/Users/sophie/Dropbox (The Francis Crick)/Sophie/Astra_project/Repositories/tcga_exp")
output_path <- "data/sophie/"

# Function definition ##########################################################
findIntersection <- function(dataframes){
  #' creates list of patients that are common between dataframes. 
  #' Requires column called "cases.submitter_id" with patient IDs
 
  list <- vector(mode = "list", length = length(dataframes)) 
  
  for (i in 1:length(dataframes)){
    df <- dataframes[[i]] 
    list[[i]] <- df$cases.submitter_id
  }
  
  # find intersection between patient lists
  patient_list <- Reduce(intersect, list)
  
  patient_list <- data.frame("cases.submitter_id" = patient_list)
  
  return(patient_list)
  
}


addMutantStatus <- function(patient_list, mut_df, gene){
  #' Adds column with mutation status to patient_list df
  #' @param patient_list a dataframe with list of patient IDs
  #' @param mut_df a dataframe with mutation information. Needs to contain a column names "cases.submitter_id" with patient IDs
  #' @param gene a string of the name of the gene of interest

  list_mut <- mut_df %>% 
    filter(Hugo_Symbol == gene) %>% 
    pull(cases.submitter_id) %>% 
    unique()
  
  patient_list_df <- data.frame(cases.submitter_id = patient_list) %>% 
    mutate(mutation_status = case_when(cases.submitter_id %in% list_mut ~ 1,
                             TRUE ~ 0))
  
  return(patient_list_df)
  
}


# Load dataframes and add necessary columns ####################################
luad_rna <- read_csv("data/sophie/luad_rna_gdc.csv")
luad_mut <- read_csv("data/sophie/luad_mut_mc3.csv")
luad_wsi <- read_csv("data/tcga-dataset/LUAD.csv")

luad_wsi <- luad_wsi[,1] %>% 
  rename("barcode" = "0") %>% 
  mutate(cases.submitter_id = substr(barcode, 21, 32))

# Run function and save output #################################################
dataframes <- list(luad_rna, luad_mut, luad_wsi)

patient_list <- findIntersection(dataframes)

patient_list <- addMutantStatus(patient_list, luad_mut, "STK11")

write_csv(patient_list, paste(output_path, "luad_patient_list.csv", sep=""))

