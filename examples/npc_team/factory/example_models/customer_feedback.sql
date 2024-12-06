
    SELECT
        feedback,
        customer_id,
        timestamp
    FROM raw_customer_feedback
    WHERE LENGTH(feedback) > 10;
    