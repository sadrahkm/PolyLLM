class AutoEncoder(nn.Module):
    def __init__(self, size_x, size_y):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # nn.Linear(size_x, 1024),
            # nn.Dropout(0.2),
            # nn.GELU(),
            # nn.BatchNorm1d(1024),
            nn.Linear(size_x, 512),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, size_y),
            nn.Dropout(0.2),
            # nn.Dropout(0.2),
            # nn.GELU(),
            # nn.BatchNorm1d(1024),
            # nn.Linear(1024, size_y),
            # nn.Dropout(0.2),
        )

    def forward(self, x):
        e0 = self.encoder(x)
        d0 = self.decoder(e0)
        return e0, d0
