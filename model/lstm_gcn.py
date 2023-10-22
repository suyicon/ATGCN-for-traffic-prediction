import torch
from torch_geometric.nn import GCNConv


class LSTMGCN(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
    ):
        super(LSTMGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._create_parameters_and_layers()

    def _create_input_gate_parameters_and_layers(self):

        self.conv_i = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_i = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_forget_gate_parameters_and_layers(self):

        self.conv_f = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_f = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_output_gate_parameters_and_layers(self):

        self.conv_o = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_o = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_memory_parameters_and_layers(self):

        self.conv_c_t = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )

        self.linear_c_t = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):

        self.conv_c = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )


    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()
        self._create_candidate_memory_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _set_candidate_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return C

    def _calculate_input_gate(self, X, edge_index, edge_weight, H):
        I = torch.cat([self.conv_i(X, edge_index, edge_weight), H], axis=1)
        I = self.linear_i(I)
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H):
        F = torch.cat([self.conv_f(X, edge_index, edge_weight), H], axis=1)
        F = self.linear_f(F)
        F = torch.sigmoid(F)
        return F

    def _calculate_memory_state(self, X, edge_index, edge_weight, H):
        C_tilde = torch.cat([self.conv_f(X, edge_index, edge_weight), H], axis=1)
        C_tilde = self.linear_c_t(C_tilde)
        C_tilde = torch.tanh(C_tilde)
        return C_tilde

    def _calculate_output_gate(self, X, edge_index, edge_weight, H):
        O = torch.cat([self.conv_o(X, edge_index, edge_weight), H], axis=1)
        O = self.linear_o(O)
        O = torch.sigmoid(O)
        return O

    def _calculate_candidate_state(self, I, F, C_tilde, C):
        C = C_tilde * I +  F * C
        return C

    def _calculate_hidden_state(self, O , C):
        C = torch.tanh(C)
        H = O*C
        return H


    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        C = self._set_hidden_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_weight, H)
        F = self._calculate_forget_gate(X, edge_index, edge_weight, H)
        O = self._calculate_output_gate(X, edge_index, edge_weight, H)
        C_tilde = self._calculate_memory_state(X, edge_index, edge_weight, H)
        C = self._calculate_candidate_state(I, F, C_tilde, C)
        H = self._calculate_hidden_state(O,C)
        return H,C


