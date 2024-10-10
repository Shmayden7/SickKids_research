# Load required libraries
library(ggplot2)
library(stats)

# Function to compute EMMs and plot with error bars
lmmDprimePlot <- function(people, ids, d_primes) {
    # Constants
    
    colours <- c('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', '#ff9896')
    markers <- c('o', 's', 'D', '^', 'v', 'p', '*', 'H', 'X', '<', '>', 'x')
    domain <- 1:6
    
    for (i in 1:nrow(d_primes)) {
        condition_data <- d_primes[i, ]
        print(condition_data)

        # Create scatter plot
        plot(domain, condition_data, 
             main = "Main title",
             xlab = "X axis title", 
             ylab = "Y axis title",
             pch = markers[i], 
             col = colours[i],
             frame = FALSE)
    }
}