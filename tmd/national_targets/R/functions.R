# general helper functions ----

cut_labels <- function(breaks) {
  n <- length(breaks)
  labels <- character(n - 1)
  for (i in 1:(n - 1)) {
    labels[i] <- paste(breaks[i], breaks[i + 1] - 1, sep = "-")
  }
  return(labels)
}


cutlabs_ge <- function(breaks) {
  n <- length(breaks)
  labels <- character(n - 1)
  for (i in 1:(n - 1)) {
    paste(breaks[i], breaks[i + 1], sep = " to < ")
  }
  return(labels)
}
# cutlabs_ge(breaks)
#
# breaks <- c(0, 1000, 2000)
# i <- 2
# paste(breaks[i], breaks[i + 1], sep = " to < ")

ht <- function(df, nrecs = 6) {
  print(utils::head(df, nrecs))
  print(utils::tail(df, nrecs))
}


lcnames <- function(df) {
  vnames <- stringr::str_to_lower(names(df))
  stats::setNames(df, vnames)
}


ns <- function(df) {
  names(df) |> sort()
}

# excel helpers for reading the target recipes xlsx file and individual xls* files ----
xlcols <- function(n) {
  # create a vector of letters in the order that Excel uses

  # a helper function that allows us to put letter column names on a dataframe
  #   that was read from an Excel file

  # usage:
  #   xlcols(53)
  #   gets the letters for the first 53 columns in a spreadsheet
  # only good for 1- and 2-letter columns, or 26 + 26 x 26 = 702 columns
  xl_letters <- c(
    LETTERS,
    sapply(LETTERS, function(x) paste0(x, LETTERS, sep = ""))
  )
  return(xl_letters[1:n])
}


get_rowmap <- function(tab) {
  # reads the target recipes xlsx file to
  # get start and end row for key data for each year of a particular IRS spreadsheet
  # from its associated mapping tab in the recipes file
  # assumes DATADIR and targfn (targets filename) are in the environment
  sheet <- paste0(tab, "_map")
  read_excel(
    path(DATADIR, targfn),
    sheet = sheet,
    range = cellranger::cell_rows(1:3)
  ) |>
    pivot_longer(-rowtype, values_to = "xlrownum") |>
    separate_wider_delim(name, delim = "_", names = c("datatype", "year")) |>
    mutate(
      table = tab,
      year = as.integer(year),
      xlrownum = as.integer(xlrownum)
    ) |>
    select(table, datatype, year, rowtype, xlrownum) |>
    arrange(table, year, datatype, desc(rowtype))
}


get_colmap <- function(tab) {
  # reads the target_recipes.xlsx file to
  # get columns of interest for each year of a particular IRS spreadsheet,
  # from its associated mapping tab in the recipes file

  # assumes DATADIR, targfn (targets filename), and allcols are in the environment
  sheet <- paste0(tab, "_map")
  col_map <- read_excel(path(DATADIR, targfn), sheet = sheet, skip = 3) |>
    pivot_longer(
      -c(vname, description, units, notes),
      values_to = "xlcolumn"
    ) |>
    separate_wider_delim(name, delim = "_", names = c("datatype", "year")) |>
    mutate(
      table = tab,
      year = as.integer(year),
      xl_colnumber = match(xlcolumn, allcols)
    ) |>
    select(
      table,
      datatype,
      year,
      xl_colnumber,
      xlcolumn,
      vname,
      description,
      units,
      notes
    ) |>
    filter(!is.na(xlcolumn), !is.na(vname)) |>
    arrange(table, datatype, year, xl_colnumber)
  col_map
}

# allcols <- xlcols(400); get_colmap("tab11")
