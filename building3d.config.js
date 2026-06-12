module.exports = {
  apps: [
    {
      name: '<Building3DEval_LiDARML_20260612>',
      script: 'scripts/run_building3d_eval.sh',
      interpreter: '/bin/bash', // converts Building3D wireframes then runs 3 external-validation eval rows
      autorestart: false, // Disable automatic restart on crashes or errors
      max_restarts: 0, // Prevent PM2 from restarting the process
      watch: false, // Enable file watching
      ignore_watch: ['data', 'logs', 'model'], // Ignore changes to these directories
      min_uptime: "5s",   // Set minimum uptime (optional)
      stop_exit_codes: [0, 1] // Ignore clean exit codes and Kill the process if it exits with error code 1
    },
  ],
};
