trainer.test(model=best_model,
            test_loader=test_data,
            test_metrics_list=DetectionMetrics_050(score_thres=0.1, 
                                                   top_k_predictions=300, 
                                                   num_cls=len(dataset_params['classes']), 
                                                   normalize_targets=True, 
                                                   post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, 
                                                                                                          nms_top_k=1000, 
                                                                                                          max_predictions=300,                                                                              
                                                                                                          nms_threshold=0.7)))


model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

img = cv2.imread('/home/panyue/AL/data/test/images/00002.jpg')
#img = cv2.cvtColor(img, cv2.Color_BGR2RGB)

outputs = best_model.predict(img)
print(outputs)

outputs.show()