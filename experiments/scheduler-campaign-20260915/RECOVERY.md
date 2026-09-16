# Arm6 startup recovery

The first arm6 startup failed before loading on rank0 because free disk space was90,533,888 bytes below the configured48GiB cache plus8GiB reserve. Rank1 was stopped by the service lifecycle. No measured arm6 window began.

Five verified full historical campaign archives were copied to rank1, SHA256 verified, and removed from rank0. Exact destinations and hashes are in archive-relocations.json. Original per-arm compressed evidence remains on rank0. Neither cache capacity nor reserve policy changed.

The controller was confirmed exited, then resumed from arm6 using baseline configs and requiring completed arms1–5. Failed arm6 configuration and status retained in failed-arm6-startup-before-measurement and arm6-startup-failure.json. New controller PID2849222; GPU watcher2849223 appends to existing half-second CSVs. The archive move and restart gap occur outside measured windows.
