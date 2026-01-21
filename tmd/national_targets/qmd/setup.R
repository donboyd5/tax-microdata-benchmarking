# QROOT <- here::here("tmd", "national_targets")

TMDINPUT <- fs::path(GITROOT, "tmd", "storage", "input")
DATADIR <- fs::path(QROOT, "..", "data")
targfn <- "target_recipes_v3.xlsx"

source(fs::path(QROOT, "R", "libraries.R"))
source(fs::path(QROOT, "R", "functions_helpers.R"))
source(fs::path(QROOT, "R", "functions_excel.R"))
