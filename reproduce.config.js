module.exports = {
  apps: [
    {
      name: '<ReproduceAllEvals_LiDARML_20260612>',
      script: 'scripts/run_all_evals.sh',
      interpreter: '/bin/bash', // sequences every eval wrapper through the deterministic pipeline
      autorestart: false, // Disable automatic restart on crashes or errors
      max_restarts: 0, // Prevent PM2 from restarting the process
      watch: false, // Enable file watching
      ignore_watch: ['data', 'logs', 'model'], // Ignore changes to these directories
      min_uptime: "5s",   // Set minimum uptime (optional)
      stop_exit_codes: [0, 1] // Ignore clean exit codes and Kill the process if it exits with error code 1
    },
  ],
};
