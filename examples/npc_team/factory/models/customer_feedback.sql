-- name: customer_feedback
-- description: Processed customer feedback data
-- source: raw_customer_feedback

SELECT
    feedback,
    customer_id,
    timestamp
FROM raw_customer_feedback
WHERE LENGTH(feedback) > 10
limit 10;
