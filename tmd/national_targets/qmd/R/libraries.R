libs <- function() {
  library(tidyverse)
  tprint <- 75 # default tibble print
  options(tibble.print_max = tprint, tibble.print_min = tprint) # show up to tprint rows

  library(readxl)
  library(vroom)
  library(fs)
  library(openxlsx2)
  library(arrow)
  library(jsonlite)
  library(tidyjson)

  library(skimr)
  library(janitor)
  library(gtExtras)
  library(DT)
  library(htmltools)
  library(kableExtra)
  library(knitr)

  library(skimr)
  library(Hmisc)
  library(gt)
  # library(btools)
}

suppressPackageStartupMessages(libs())

# pkgs <- c(
#   "tidyverse",
#   "readxl",
#   "vroom",
#   "fs",
#   "openxlsx2",
#   "arrow",
#   "jsonlite",
#   "tidyjson",
#   "skimr",
#   "janitor",
#   "gtExtras",
#   "DT",
#   "htmltools",
#   "kableExtra",
#   "knitr",
#   "Hmisc",
#   "gt"
# )

# url <- "https://cran.r-project.org/src/contrib/xml2_1.5.1.tar.gz"
# install.packages("https://cran.r-project.org/src/contrib/xml2_1.5.1.tar.gz", repos = NULL, type = "source")
# devtools::install_github("colearendt/tidyjson")
# pkgs <- c("tidyjson")
# install.packages(pkgs)
