module.exports = {
  apps: [
    {
      name: '<Model_training_LiDARML_03122025>',
      script: 'trainModel.py',
      interpreter: '/home/ubuntu/miniconda3/envs/LiDARML/bin/python', // Path to python interpreter
      autorestart: false, // Disable automatic restart on crashes or errors
      max_restarts: 0, // Prevent PM2 from restarting the process
      // stop_exit_codes: [1], // Kill the process if it exits with error code 1
      watch: false, // Enable file watching
      ignore_watch: ['data', 'logs', 'model'], // Ignore changes to these directories
      min_uptime: "5s",   // Set minimum uptime (optional)
      stop_exit_codes: [0, 1] // Ignore clean exit codes and Kill the process if it exits with error code 1
    },
  ],
};