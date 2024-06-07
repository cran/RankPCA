#' Calculate Variable Ranking
#'
#' This function calculates the ranking of variables based on the sum of absolute values for each row of loading vectors.
#'
#' @param loading_vectors A matrix containing loading vectors.
#' @return A data frame containing the ranked variables.
#' @examples
#' # Define row and column names
#' row_names <- c("Category1A", "Category1B", "Category1C", "Category2X", "Category2Y",
#'                "Category2Z", "Category3M", "Category3N", "Category3O", "Continuous1",
#'                "Continuous2", "Continuous3", "Continuous4", "Continuous5")
#'
#' col_names <- paste0("PC", 1:8)
#'
#' # Define the data matrix
#' loading_vectors <- matrix(c(
#'   0.199, 0.268, 0.189, 0.641, 0.092, 0.171, 0.079, -0.070,
#'   0.244, -0.371, 0.042, -0.426, 0.358, -0.070, 0.016, 0.371,
#'   -0.435, 0.099, -0.227, -0.216, -0.441, -0.100, -0.094, -0.294,
#'   0.087, -0.338, 0.458, 0.083, -0.515, -0.150, 0.007, 0.029,
#'   -0.473, 0.170, -0.164, 0.172, 0.296, 0.006, -0.044, 0.462,
#'   0.407, 0.155, -0.279, -0.261, 0.198, 0.141, 0.039, -0.510,
#'   0.101, -0.487, -0.465, 0.302, -0.117, 0.062, 0.036, 0.035,
#'   0.145, 0.546, 0.057, -0.211, -0.123, -0.325, 0.287, 0.191,
#'   -0.274, -0.003, 0.491, -0.134, 0.271, 0.272, -0.349, -0.245,
#'   0.290, 0.207, 0.001, -0.048, -0.250, -0.090, -0.275, 0.330,
#'   -0.134, 0.099, -0.277, -0.072, -0.180, 0.485, 0.134, 0.147,
#'   0.006, 0.051, -0.216, 0.007, 0.008, -0.278, -0.712, 0.004,
#'   0.320, 0.145, -0.061, 0.146, -0.078, 0.215, -0.414, 0.096,
#'   0.061, 0.044, 0.096, -0.271, -0.273, 0.603, -0.064, 0.245
#' ), ncol = 8, byrow = TRUE)
#'
#' # Assign row and column names
#' rownames(loading_vectors) <- row_names
#' colnames(loading_vectors) <- col_names
#'
#' # Now you can use the loading_vectors variable in your code
#' print(loading_vectors)
#' # rank the variables
#' ranked_variables <- variable_ranking(loading_vectors)
#' print(ranked_variables)
#' @export
variable_ranking <- function(loading_vectors) {
  # Calculate the sum of absolute values for each row
  row_sums <- rowSums(abs(loading_vectors))

  # Combine row sums with variable names
  variable_sums <- data.frame(variable = rownames(loading_vectors), contribution = row_sums)

  # Rank the variables based on their sums
  ranked_variables <- variable_sums[order(variable_sums$contribution, decreasing = TRUE), ]

  # Return the ranked variables
  return(ranked_variables)
}
