# excel helpers for reading the target recipes xlsx file and individual xls* files ----

xlcol_to_num <- function(s) {
  # convert vector of Excel column labels s such as c("BF", "AG") to column numbers
  # created with help from AI
  sapply(strsplit(toupper(s), "", fixed = TRUE), \(chars) {
    Reduce(\(a, b) a * 26 + b, match(chars, LETTERS)) |> as.integer()
  })
}

xlnum_to_col <- function(n) {
  # convert vector of Excel column numbers such as c(58, 33) to column labels
  unname(sapply(n, \(x) {
    out <- character()
    while (x > 0) {
      x <- x - 1L
      out <- c(LETTERS[x %% 26 + 1L], out)
      x <- x %/% 26
    }
    paste0(out, collapse = "")
  }))
}

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
# allcols <- xlcols(400); get_colmap("tab11")
