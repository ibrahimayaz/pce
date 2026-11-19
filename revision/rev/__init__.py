"""Revision pipeline for manuscript Access-2026-47475.

All preprocessing (zero-variance removal, standardisation, mRMR) and all
hyper-parameter tuning are performed strictly inside the training part of every
outer cross-validation fold (nested CV).  Every unit of work is a *job* whose
result is written atomically to disk, so an interrupted run (power loss,
reboot, Ctrl+C) can be resumed without losing finished work.
"""
