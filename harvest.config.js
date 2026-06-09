module.exports = {
  apps: [
    {
      name: '<MergeHarvest_LiDARML_20260610_fold_0>',
      script: 'scripts/harvest_fold.py',
      args: '--fold 0 --epochs 50 --output-dir artifacts/merge_classifier',
      interpreter: '/home/ubuntu/miniconda3/envs/LiDARML/bin/python', // Path to python interpreter
      env: { MKL_THREADING_LAYER: 'GNU' }, // avoid sklearn/torch OpenMP symbol clash
      autorestart: false, // Disable automatic restart on crashes or errors
      max_restarts: 0, // Prevent PM2 from restarting the process
      watch: false, // Enable file watching
      ignore_watch: ['data', 'logs', 'model'], // Ignore changes to these directories
      min_uptime: "5s",   // Set minimum uptime (optional)
      stop_exit_codes: [0, 1] // Ignore clean exit codes and Kill the process if it exits with error code 1
    },
    {
      name: '<MergeHarvest_LiDARML_20260610_fold_1>',
      script: 'scripts/harvest_fold.py',
      args: '--fold 1 --epochs 50 --output-dir artifacts/merge_classifier',
      interpreter: '/home/ubuntu/miniconda3/envs/LiDARML/bin/python',
      env: { MKL_THREADING_LAYER: 'GNU' },
      autorestart: false,
      max_restarts: 0,
      watch: false,
      ignore_watch: ['data', 'logs', 'model'],
      min_uptime: "5s",
      stop_exit_codes: [0, 1]
    },
    {
      name: '<MergeHarvest_LiDARML_20260610_fold_2>',
      script: 'scripts/harvest_fold.py',
      args: '--fold 2 --epochs 50 --output-dir artifacts/merge_classifier',
      interpreter: '/home/ubuntu/miniconda3/envs/LiDARML/bin/python',
      env: { MKL_THREADING_LAYER: 'GNU' },
      autorestart: false,
      max_restarts: 0,
      watch: false,
      ignore_watch: ['data', 'logs', 'model'],
      min_uptime: "5s",
      stop_exit_codes: [0, 1]
    },
    {
      name: '<MergeHarvest_LiDARML_20260610_fold_3>',
      script: 'scripts/harvest_fold.py',
      args: '--fold 3 --epochs 50 --output-dir artifacts/merge_classifier',
      interpreter: '/home/ubuntu/miniconda3/envs/LiDARML/bin/python',
      env: { MKL_THREADING_LAYER: 'GNU' },
      autorestart: false,
      max_restarts: 0,
      watch: false,
      ignore_watch: ['data', 'logs', 'model'],
      min_uptime: "5s",
      stop_exit_codes: [0, 1]
    },
    {
      name: '<MergeHarvest_LiDARML_20260610_fold_4>',
      script: 'scripts/harvest_fold.py',
      args: '--fold 4 --epochs 50 --output-dir artifacts/merge_classifier',
      interpreter: '/home/ubuntu/miniconda3/envs/LiDARML/bin/python',
      env: { MKL_THREADING_LAYER: 'GNU' },
      autorestart: false,
      max_restarts: 0,
      watch: false,
      ignore_watch: ['data', 'logs', 'model'],
      min_uptime: "5s",
      stop_exit_codes: [0, 1]
    },
  ],
};
