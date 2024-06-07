#' Rank Principal Component Analysis for Mixed Data Types
#'
#' This function performs Principal Component Analysis (PCA) on datasets containing both categorical and continuous variables. It facilitates data preprocessing, encoding of categorical variables, and computes PCA to determine the optimal number of principal components based on a specified variance threshold. The function also computes composite indices for ranking observations.
#'
#' @param data data to be analyzed.
#' @param range_cat_var Range of categorical variables.
#' @param range_continuous_var Range of continuous variables.
#' @param threshold Threshold for cumulative variance explained.
#' @return A list containing PCA results and composite index.
#' @examples
#' # Create a sample dataset
#' set.seed(123)
#' sample_data <- data.frame(
#'   Category1 = sample(c("A", "B", "C"), 100, replace = TRUE),
#'   Category2 = sample(c("X", "Y", "Z"), 100, replace = TRUE),
#'   Category3 = sample(c("M", "N", "O"), 100, replace = TRUE),
#'   Continuous1 = rnorm(100),
#'   Continuous2 = runif(100, min = 0, max = 100),
#'   Continuous3 = rnorm(100, mean = 50, sd = 10),
#'   Continuous4 = rpois(100, lambda = 5),
#'   Continuous5 = rbinom(100, size = 10, prob = 0.5)
#' )
#' result <- rankPCA(data = sample_data,
#'                   range_cat_var = 1:3,
#'                   range_continuous_var = 4:8,
#'                   threshold = 80)
#'
#' # Access the results
#' eigenvalues_pca <- result$eigenvalues_pca
#' pca_max_dim <- result$pca_max_dim
#' coordinates <- result$coordinates
#' eigenvalues <- result$eigenvalues
#' weighted_coordinates <- result$weighted_coordinates
#' weighted_sums <- result$weighted_sums
#' composite_index <- result$composite_index
#' loading_vectors <- result$loading_vectors
#' @references
#' Garai, S., & Paul, R. K. (2023). Development of MCS based-ensemble models using CEEMDAN decomposition and machine intelligence. Intelligent Systems with Applications, 18, 200202, https://doi.org/10.1016/j.iswa.2023.200202.
#' @importFrom stats prcomp sd var predict
#' @importFrom caret dummyVars
#' @export
rankPCA <- function(data, range_cat_var, range_continuous_var, threshold) {
  # Separate continuous and categorical variables
  categorical_vars <- data[, range_cat_var]
  continuous_vars <- data[, range_continuous_var]

  # Convert categorical variables to factors
  categorical_vars <- apply(categorical_vars, 2, as.factor)

  cat_vars <- colnames(data[, range_cat_var])

  # Assuming 'data' is your dataset and 'cat_vars' is a vector of categorical variable names
  dummy_data <- dummyVars("~.", data = data[, range_cat_var], fullRank = TRUE)
  encoded_data <- predict(dummy_data, newdata = data)

  final_data <- cbind(encoded_data, continuous_vars)

  # Standardize continuous variables
  standardized_continuous <- scale(final_data)

  # Perform PCA on continuous variables
  pca_result_continuous <- prcomp(standardized_continuous)

  # Access the eigenvalues from the prcomp result
  eigenvalues <- pca_result_continuous$sdev^2

  # Calculate the proportion of variance explained
  variance_explained <- eigenvalues / sum(eigenvalues)

  # Calculate the cumulative proportion of variance explained
  cumulative_variance_explained <- cumsum(variance_explained)

  # Create a data frame with eigenvalues, variance explained, and cumulative variance explained
  eigenvalues_pca <- data.frame(
    Dimension = 1:length(eigenvalues),
    eigenvalue = eigenvalues,
    percentage_variance = variance_explained * 100,
    cumulative_percentage_variance = cumulative_variance_explained * 100
  )

  # Find row numbers where the last column's value is exactly greater than the threshold for PCA
  rows_above_threshold_pca <- which(eigenvalues_pca$cumulative_percentage_variance > threshold)
  pca_max_dim <- rows_above_threshold_pca[1]

  # Create column names for PCA components
  pca_column_names <- paste0("PC", 1:pca_max_dim)

  # Combine results
  combined_result <- as.data.frame(pca_result_continuous$x)

  coordinates <- data.frame(
    combined_result[, pca_column_names][1:pca_max_dim]
  )

  # Extract 1 to pca_max_dim rows values from pca_max_dim$eigenvalue
  eigenvalues <- eigenvalues_pca[1:pca_max_dim, "eigenvalue"]

  # Calculate the weighted sum of coordinates using corresponding eigenvalues for each row
  weighted_coordinates <- coordinates * sqrt(eigenvalues)

  # Calculate the weighted sums for each column
  weighted_sums <- rowSums(weighted_coordinates)

  composite_index <- weighted_sums / sum(sqrt(eigenvalues))

  # Add the columns to the original data or do any other further processing
  result <- cbind(coordinates, weighted_sums, composite_index)

  # Get the loading vectors from PCA result
  loading_vectors <- pca_result_continuous$rotation[, 1:pca_max_dim]

  # Round the other columns to three decimal places
  loading_vectors <- round(loading_vectors, 3)

  # Return the relevant results
  return(list(
    eigenvalues_pca = eigenvalues_pca,
    pca_max_dim = pca_max_dim,
    coordinates = coordinates,
    eigenvalues = eigenvalues,
    weighted_coordinates = weighted_coordinates,
    weighted_sums = weighted_sums,
    composite_index = composite_index,
    loading_vectors = loading_vectors
  ))
}
