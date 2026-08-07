# Fig 5d: lung-infection tAge trajectory meta-regression (LUNG half); AD half is an
# unimplemented gap (see bottom of this file).
#
# LUNG HALF -- REAL, ported from v_pipeline/MixedEffectsModel.R with two bug fixes applied
# (pre-authorized by the author; not a silent behavior change -- both are described inline):
#
#   (1) The original discarded its own `read.csv()` result on the very next line by
#       reassigning `data <- lung_infection_df_preds`, an object assumed already present in
#       the interactive R session. This made the script non-reproducible standalone (it would
#       error with "object 'lung_infection_df_preds' not found" if run fresh). Fixed by
#       actually using the `read.csv()` result.
#   (2) The original never saved any output -- every plot was only shown interactively
#       (`ggplot(...)` printed to a device, no `ggsave()` anywhere). Fixed by adding `ggsave()`
#       calls for the final summary table and the final trajectory plot.
#
# Every statistical/modeling choice is otherwise UNCHANGED: this is `metafor::rma.uni(method
# = "REML")` inverse-variance meta-REGRESSION on per-sample (file x time x norm x condition)
# summary means/variances -- NOT a true mixed-effects model (no `lme4::lmer`, no
# random-intercept-per-animal term) despite the original filename. This discrepancy was
# already flagged during the code audit (stAge-release/INVENTORY.md) and is preserved as-is
# here, not silently "corrected" to a real mixed model -- that would be a reanalysis, not a
# refactor.
#
# The original file also contained ~8 near-duplicate exploratory plot iterations (progressively
# adding raw-mean overlays, SE bars, etc. to the same fitted-trajectory plot) -- consolidated
# here to the single final/most-complete iteration ("Transcriptomic Aging Trajectories by Group
# and Clock", the time*condition interaction model), consistent with how iterative-editing
# residue was handled elsewhere in this release (e.g. celltype_hotspot_tAge.ipynb's region
# analysis, which also kept only the final of several re-executed iterations).

suppressPackageStartupMessages({
  library(metafor)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

#' Run the lung-infection tAge meta-regression and save the summary table + trajectory plot.
#'
#' @param csv_path Path to the per-sample tAge predictions CSV (columns: file, time, norm,
#'   condition, age -- one row per spot/sample; `condition` in {"Young","Aged"}, `norm` in
#'   {"tAge_SM","tAge_YM"}, `time` = days/months post-infection).
#' @param save_dir Directory to write outputs into (created if missing).
run_fig5d_lung <- function(csv_path, save_dir = ".") {
  dir.create(save_dir, showWarnings = FALSE, recursive = TRUE)

  # BUG FIX (1): use the actual CSV read, not an assumed-preloaded object.
  data <- read.csv(csv_path)

  # Per-sample (file x time x norm x condition) summary stats for inverse-variance weighting.
  meta_df <- data %>%
    group_by(file, time, norm, condition) %>%
    summarise(
      yi = mean(age),
      vi = var(age) / n(),
      n = n(),
      .groups = "drop"
    )

  # Per (clock, condition) simple meta-regression: age ~ time.
  results_list <- list()
  for (clock in c("tAge_SM", "tAge_YM")) {
    for (cond in c("Young", "Aged")) {
      sub_df <- meta_df %>% filter(norm == clock, condition == cond)
      if (nrow(sub_df) >= 3) {
        res <- rma.uni(yi = yi, vi = vi, mods = ~time, data = sub_df, method = "REML")
        results_list[[paste(clock, cond, sep = "_")]] <- data.frame(
          clock = clock, condition = cond,
          intercept = res$b[1, 1], slope = res$b[2, 1],
          intercept_se = res$se[1], slope_se = res$se[2],
          intercept_p = res$pval[1], slope_p = res$pval[2]
        )
      } else {
        message(sprintf("Skipped %s / %s: insufficient data (n=%d rows)", clock, cond, nrow(sub_df)))
      }
    }
  }
  results <- bind_rows(results_list) %>%
    mutate(
      group = factor(paste(clock, condition),
                      levels = c("tAge_SM Young", "tAge_SM Aged", "tAge_YM Young", "tAge_YM Aged")),
      ci_lower = slope - 1.96 * slope_se,
      ci_upper = slope + 1.96 * slope_se
    )

  # Primary test: does the aging slope differ between Young and Aged, per clock?
  # (time * condition interaction model -- this is the key statistical result for Fig 5d.)
  interaction_list <- list()
  for (clock in c("tAge_SM", "tAge_YM")) {
    sub_df <- meta_df %>%
      filter(norm == clock) %>%
      mutate(condition = factor(condition, levels = c("Young", "Aged")))
    if (nrow(sub_df) >= 3) {
      res <- rma.uni(yi = yi, vi = vi, mods = ~ time * condition, data = sub_df, method = "REML")
      interaction_list[[clock]] <- data.frame(
        clock = clock,
        intercept = res$b[1, 1], time_coef = res$b[2, 1],
        cond_coef = res$b[3, 1], interaction = res$b[4, 1],
        interaction_se = res$se[4], interaction_p = res$pval[4]
      )
    }
  }
  interaction_results <- bind_rows(interaction_list)

  write.csv(results, file.path(save_dir, "fig5d_lung_meta_regression_per_condition.csv"), row.names = FALSE)
  write.csv(interaction_results, file.path(save_dir, "fig5d_lung_meta_regression_interaction.csv"), row.names = FALSE)

  # Final trajectory plot: time*condition interaction model, predicted trajectories per clock.
  fitted <- expand.grid(
    time = c(0, 3, 9),
    condition = c("Young", "Aged"),
    clock = unique(interaction_results$clock)
  ) %>%
    left_join(interaction_results, by = "clock") %>%
    mutate(
      condition = factor(condition, levels = c("Young", "Aged")),
      predicted = case_when(
        condition == "Young" ~ intercept + time_coef * time,
        condition == "Aged"  ~ intercept + time_coef * time + interaction * time
      )
    )

  p <- ggplot(fitted, aes(x = time, y = predicted, color = condition, linetype = clock)) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 2.5) +
    labs(
      title = "Transcriptomic Aging Trajectories by Group and Clock",
      x = "Infection Timepoint (days)",
      y = "Predicted Transcriptomic Age"
    ) +
    scale_color_manual(values = c("Young" = "#1f77b4", "Aged" = "#ff7f0e")) +
    theme_minimal(base_size = 14)

  # BUG FIX (2): actually save the figure (original only ever called ggplot() interactively).
  ggsave(file.path(save_dir, "fig5d_lung_trajectories.pdf"), plot = p, width = 7, height = 5)

  list(per_condition = results, interaction = interaction_results, plot = p)
}

# ---- AD component: TODO(gap) ----
# No AD-specific mixed-effects/meta-regression code was found anywhere in the audited
# codebase (v_pipeline, the dataset-specific pipeline directories, or the top-level R
# scripts). `rds_to_h5ad.R` (stAge-release/R/rds_to_h5ad.R) does contain a live Seurat-RDS
# conversion block for an AD-human dataset, discovered during Phase 2 -- a real, if partial,
# lead for producing the input data -- but the actual meta-regression/mixed-effects analysis
# for the AD half of this figure still needs to be authored fresh. See
# stAge-release/INVENTORY.md gap list item 5.
#
# run_fig5d_ad <- function(...) {
#   stop("AD component not implemented -- see comment block above and INVENTORY.md gap item 5.")
# }
