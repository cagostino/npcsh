                                                                                  
    SELECT                                                                                                   
        customer_id,                                                                                         
        feedback,                                                                                            
        timestamp,                                                                                           
        synthesize(                                                                                          
            "feedback text: {feedback}",                                                                     
            "../analyst.npc",                                                                                       
            "feedback_analysis"                                                                              
        ) as ai_analysis                                                                                     
    FROM {{ ref('customer_feedback') }};                                                                     
    