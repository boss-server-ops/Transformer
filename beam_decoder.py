import torch
from data_loader import subsequent_mask


class Beam:
    """实现 Beam 搜索算法"""

    def __init__(self, beam_width, pad_token, start_token, end_token, device=None):
        self.beam_width = beam_width
        self.PAD_TOKEN = pad_token
        self.START_TOKEN = start_token
        self.END_TOKEN = end_token
        self.device = device if device is not None else torch.device('cpu')
        
        # Beam 中每个序列的分数
        self.scores = torch.zeros((beam_width,), dtype=torch.float, device=self.device)
        self.all_scores = []

        # 存储每个时间步的后向指针
        self.prev_indices = []

        # 存储每个时间步的输出
        self.outputs = [torch.full((beam_width,), self.PAD_TOKEN, dtype=torch.long, device=self.device)]
        self.outputs[0][0] = self.START_TOKEN

        # 标识是否完成
        self._completed = False

    def get_current_outputs(self):
        """获取当前时间步的输出"""
        return self._gather_tentative_hypotheses()

    def get_current_backpointers(self):
        """获取当前时间步的后向指针"""
        return self.prev_indices[-1]

    @property
    def completed(self):
        return self._completed

    def advance(self, log_probs):
        """更新 Beam 的状态并检查是否完成"""
        num_tokens = log_probs.size(1)

        # 更新分数
        if len(self.prev_indices) > 0:
            beam_scores = log_probs + self.scores.unsqueeze(1).expand_as(log_probs)
        else:
            beam_scores = log_probs[0]

        flat_beam_scores = beam_scores.view(-1)
        top_scores, top_indices = flat_beam_scores.topk(self.beam_width, 0, True, True)

        self.all_scores.append(self.scores)
        self.scores = top_scores

        prev_indices = top_indices // num_tokens
        self.prev_indices.append(prev_indices)
        self.outputs.append(top_indices % num_tokens)

        # 检查最优的 Beam 是否以 <EOS> 结束
        if self.outputs[-1][0].item() == self.END_TOKEN:
            self._completed = True
            self.all_scores.append(self.scores)

        return self._completed

    def sort_by_scores(self):
        """对分数进行排序"""
        return torch.sort(self.scores, 0, True)

    def get_best_score_and_index(self):
        """获取最佳分数和对应的索引"""
        sorted_scores, sorted_indices = self.sort_by_scores()
        return sorted_scores[0], sorted_indices[0]

    def _gather_tentative_hypotheses(self):
        """获取当前时间步的解码序列"""
        if len(self.outputs) == 1:
            decoded_sequence = self.outputs[0].unsqueeze(1)
        else:
            _, sorted_indices = self.sort_by_scores()
            hypotheses = [self._trace_hypothesis(idx) for idx in sorted_indices]
            hypotheses = [[self.START_TOKEN] + h for h in hypotheses]
            decoded_sequence = torch.LongTensor(hypotheses)
        
        return decoded_sequence

    def _trace_hypothesis(self, idx):
        """回溯构造完整的假设序列"""
        hypothesis = []
        for step in range(len(self.prev_indices) - 1, -1, -1):
            hypothesis.append(self.outputs[step + 1][idx])
            idx = self.prev_indices[step][idx]
        
        return list(map(lambda x: x.item(), hypothesis[::-1]))


def beam_search(model, source, src_mask, max_length, pad_token, start_token, end_token, beam_width, device):
    """ 执行一个批次的翻译 """

    def map_instance_to_tensor_position(instance_indices):
        """将实例索引映射到张量位置"""
        return {idx: pos for pos, idx in enumerate(instance_indices)}

    def select_active_parts(tensor, active_indices, prev_active_count, beam_width):
        """选择与活跃实例相关的张量部分"""
        _, *other_dims = tensor.size()
        curr_active_count = len(active_indices)
        new_shape = (curr_active_count * beam_width, *other_dims)

        tensor = tensor.view(prev_active_count, -1)
        tensor = tensor.index_select(0, active_indices)
        tensor = tensor.view(*new_shape)

        return tensor

    def collate_active_information(encoder_output, mask, instance_position_map, active_indices):
        """收集活跃实例的信息"""
        prev_active_count = len(instance_position_map)
        active_positions = [instance_position_map[idx] for idx in active_indices]
        active_positions = torch.LongTensor(active_positions).to(device)

        active_encoder_output = select_active_parts(encoder_output, active_positions, prev_active_count, beam_width)
        active_instance_position_map = map_instance_to_tensor_position(active_indices)
        active_mask = select_active_parts(mask, active_positions, prev_active_count, beam_width)

        return active_encoder_output, active_mask, active_instance_position_map

    def decode_step(instance_beams, seq_length, encoder_output, instance_position_map, beam_width):
        """解码和更新 Beam 状态，并返回活跃 Beam 的索引"""

        def prepare_partial_sequences(instance_beams, seq_length):
            partial_sequences = [beam.get_current_outputs() for beam in instance_beams if not beam.completed]
            partial_sequences = torch.stack(partial_sequences).to(device)
            partial_sequences = partial_sequences.view(-1, seq_length)
            return partial_sequences

        def predict_next_word(seq, encoder_output, num_active_instances, beam_width):
            assert encoder_output.shape[0] == seq.shape[0] == src_mask.shape[0]
            decoder_output = model.decode(encoder_output, src_mask, seq, subsequent_mask(seq.size(1)).type_as(source.data))
            word_log_probs = model.generator(decoder_output[:, -1])
            word_log_probs = word_log_probs.view(num_active_instances, beam_width, -1)
            return word_log_probs

        def get_active_indices(instance_beams, log_probs, instance_position_map):
            active_indices = []
            for inst_idx, position in instance_position_map.items():
                is_complete = instance_beams[inst_idx].advance(log_probs[position])
                if not is_complete:
                    active_indices.append(inst_idx)
            return active_indices

        num_active_instances = len(instance_position_map)
        partial_sequences = prepare_partial_sequences(instance_beams, seq_length)
        word_log_probs = predict_next_word(partial_sequences, encoder_output, num_active_instances, beam_width)
        active_indices = get_active_indices(instance_beams, word_log_probs, instance_position_map)

        return active_indices

    def gather_hypotheses_and_scores(instance_beams, num_best):
        all_hypotheses, all_scores = [], []
        for beam in instance_beams:
            sorted_scores, top_indices = beam.sort_by_scores()
            all_scores.append(sorted_scores[:num_best])

            top_hypotheses = [beam._trace_hypothesis(i) for i in top_indices[:num_best]]
            all_hypotheses.append(top_hypotheses)
        return all_hypotheses, all_scores

    with torch.no_grad():
        encoded_source = model.encode(source, src_mask)
        
        batch_size, seq_length, hidden_dim = encoded_source.size()
        encoded_source = encoded_source.repeat(1, beam_width, 1).view(batch_size * beam_width, seq_length, hidden_dim)
        src_mask = src_mask.repeat(1, beam_width, 1).view(batch_size * beam_width, 1, src_mask.shape[-1])

        instance_beams = [Beam(beam_width, pad_token, start_token, end_token, device) for _ in range(batch_size)]
        
        active_indices = list(range(batch_size))
        instance_position_map = map_instance_to_tensor_position(active_indices)

        for step in range(1, max_length + 1):
            active_indices = decode_step(instance_beams, step, encoded_source, instance_position_map, beam_width)

            if not active_indices:
                break

            encoded_source, src_mask, instance_position_map = collate_active_information(
                encoded_source, src_mask, instance_position_map, active_indices)

    hypotheses, scores = gather_hypotheses_and_scores(instance_beams, beam_width)

    return hypotheses, scores
