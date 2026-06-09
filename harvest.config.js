module.exports = {
  apps: [
    {
      name: '<MergeHarvest_LiDARML_20260610_all_folds>',
      script: 'scripts/run_harvest.sh',
      interpreter: '/bin/bash', // bash wrapper loops folds 0..4 sequentially
      autorestart: false, // Disable automatic restart on crashes or errors
      max_restarts: 0, // Prevent PM2 from restarting the process
      watch: false, // Enable file watching
      ignore_watch: ['data', 'logs', 'model'], // Ignore changes to these directories
      min_uptime: "5s",   // Set minimum uptime (optional)
      stop_exit_codes: [0, 1] // Ignore clean exit codes and Kill the process if it exits with error code 1
    },
  ],
};
