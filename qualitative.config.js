module.exports = {
  apps: [
    {
      name: '<QualitativeDump_LiDARML_20260612>',
      script: 'scripts/dump_predictions.py',
      args: '--output-dir evaluation_results/qualitative',
      interpreter: '/home/ubuntu/miniconda3/envs/LiDARML/bin/python', // Path to python interpreter
      env: { MKL_THREADING_LAYER: 'GNU' }, // avoid sklearn/torch OpenMP symbol clash
      autorestart: false, // Disable automatic restart on crashes or errors
      max_restarts: 0, // Prevent PM2 from restarting the process
      watch: false, // Enable file watching
      ignore_watch: ['data', 'logs', 'model'], // Ignore changes to these directories
      min_uptime: "5s",   // Set minimum uptime (optional)
      stop_exit_codes: [0, 1] // Ignore clean exit codes and Kill the process if it exits with error code 1
    },
  ],
};
