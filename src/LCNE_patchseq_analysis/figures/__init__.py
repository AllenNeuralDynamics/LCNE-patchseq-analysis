# Global filter for patch-seq analysis figures

global_filter = (
    "(`jem-status_reporter` == 'Positive') & "
    "(`injection region` != 'Non-Retro') & "
    "(`injection region` != 'Thalamus')"
)
