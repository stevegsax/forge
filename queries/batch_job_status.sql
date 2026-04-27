SELECT status, COUNT(*) AS count, COUNT(error_message) AS errors
FROM batch_jobs
GROUP BY status
ORDER BY count DESC;
