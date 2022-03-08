from src.inference.model_inference import Inference

inference_model = Inference("./", encoder_name="shtoshni/longformer_coreference_ontonotes")

doc = """
Thành phố Đà Nẵng cũng là nơi đặt các văn phòng đại diện của các bộ, ban ngành của Chính phủ Việt Nam, các cơ quan của Trung ương Đảng Cộng sản Việt Nam làm nhiệm vụ phụ trách khu vực Miền Trung - Tây Nguyên, nhiều thứ hai sau Thành phố Hồ Chí Minh. Thành phố Đà Nẵng có 8 đơn vị hành chính cấp huyện, gồm 6 quận và 2 huyện. Tổng diện tích thành phố là 1285,4 km², gồm 56 đơn vị hành chính cấp xã, trong đó có 45 phường và 11 xã. Ngoại trừ quận Cẩm Lệ, năm quận còn lại của thành phố đều giáp biển.
"""

output = inference_model.perform_coreference(doc)
for cluster in output["clusters"]:
	print(cluster)
