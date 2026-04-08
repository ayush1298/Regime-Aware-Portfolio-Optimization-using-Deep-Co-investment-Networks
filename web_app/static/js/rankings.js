// static/js/rankings.js
class StockRankings {
    constructor() {
        this.currentYear = 2012;
        this.densityChart = null;
        this.marketCapChart = null;
        this.bindEvents();
        this.loadRankings(this.currentYear);
        this.renderThesisCharts();
    }

    bindEvents() {
        document.getElementById('yearSelectRanking').addEventListener('change', (e) => {
            this.currentYear = parseInt(e.target.value);
            this.loadRankings(this.currentYear);
            this.updateYearDisplay();
        });
    }

    async loadRankings(year) {
        try {
            const response = await fetch(`/api/rankings/${year}`);
            const data = await response.json();
            this.renderRankings(data);
            this.updateHitRatio(data);
        } catch (error) {
            console.error('Error loading rankings:', error);
        }
    }

    renderRankings(data) {
        const tbody = document.getElementById('rankingsBody');
        tbody.innerHTML = '';

        data.top_stocks.forEach((stock, index) => {
            const row = document.createElement('tr');
            const isPredicted = data.deepcnl_predictions.includes(stock.ticker);
            
            if (isPredicted) {
                row.classList.add('predicted-stock');
            }

            row.innerHTML = `
                <td>${index + 1}</td>
                <td><strong>${stock.ticker}</strong></td>
                <td>${stock.company_name}</td>
                <td>${stock.return.toFixed(2)}%</td>
                <td>
                    ${isPredicted ? 
                        '<i class="fas fa-check-circle text-success"></i> Predicted' : 
                        '<i class="fas fa-times-circle text-muted"></i> Not Predicted'
                    }
                </td>
            `;

            tbody.appendChild(row);
        });
    }

    updateHitRatio(data) {
        const hitRatio = data.hit_ratio;
        const hitRatioValue = document.getElementById('hitRatioValue');
        hitRatioValue.textContent = `${data.deepcnl_hits} / 10`;
        
        // Add animation
        hitRatioValue.style.transform = 'scale(1.1)';
        setTimeout(() => {
            hitRatioValue.style.transform = 'scale(1)';
        }, 200);
    }

    updateYearDisplay() {
        document.getElementById('selectedYear').textContent = this.currentYear;
        document.getElementById('rankingYear').textContent = this.currentYear;
    }

    // ====== Benchmark Charts ======

    renderThesisCharts() {
        this.renderDensityChart();
        this.renderMarketCapChart();
    }

    renderDensityChart() {
        const ctx = document.getElementById('densityChart');
        if (!ctx) return;

        // Data reverse-engineered from the thesis screenshots
        const labels = ['Top 10', 'Top 20', 'Top 30'];
        const deepCNLData = [0.45, 0.39, 0.30];
        const pccData      = [0.31, 0.40, 0.39];

        this.densityChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'DeepCNL',
                        data: deepCNLData,
                        borderColor: 'rgba(52, 152, 219, 1)',
                        backgroundColor: 'rgba(52, 152, 219, 0.15)',
                        fill: true,
                        tension: 0.35,
                        pointRadius: 6,
                        pointHoverRadius: 9,
                        pointBackgroundColor: 'rgba(52, 152, 219, 1)',
                        borderWidth: 3
                    },
                    {
                        label: 'PCC',
                        data: pccData,
                        borderColor: 'rgba(231, 76, 60, 1)',
                        backgroundColor: 'rgba(231, 76, 60, 0.10)',
                        fill: true,
                        tension: 0.35,
                        pointRadius: 6,
                        pointHoverRadius: 9,
                        pointBackgroundColor: 'rgba(231, 76, 60, 1)',
                        borderWidth: 3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            usePointStyle: true,
                            padding: 20,
                            font: { size: 13 }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(ctx) {
                                return `${ctx.dataset.label}: ${(ctx.parsed.y * 100).toFixed(1)}%`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 0.55,
                        ticks: {
                            callback: v => `${(v * 100).toFixed(0)}%`,
                            font: { size: 12 }
                        },
                        title: {
                            display: true,
                            text: 'Investment Density',
                            font: { size: 13, weight: 'bold' }
                        },
                        grid: { color: 'rgba(0,0,0,0.06)' }
                    },
                    x: {
                        ticks: { font: { size: 12 } },
                        grid: { display: false }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }

    renderMarketCapChart() {
        const ctx = document.getElementById('marketCapChart');
        if (!ctx) return;

        const labels = ['DeepCNL', 'PCC'];
        const dataValues = [223, 68]; // in billions

        this.marketCapChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Avg. Market Cap ($B)',
                    data: dataValues,
                    backgroundColor: [
                        'rgba(52, 152, 219, 0.8)',
                        'rgba(231, 76, 60, 0.8)'
                    ],
                    borderColor: [
                        'rgba(52, 152, 219, 1)',
                        'rgba(231, 76, 60, 1)'
                    ],
                    borderWidth: 2,
                    borderRadius: 6,
                    barPercentage: 0.55
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(ctx) {
                                return `$${ctx.parsed.y}B`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 280,
                        ticks: {
                            callback: v => `$${v}B`,
                            font: { size: 12 }
                        },
                        title: {
                            display: true,
                            text: 'Average Market Cap',
                            font: { size: 13, weight: 'bold' }
                        },
                        grid: { color: 'rgba(0,0,0,0.06)' }
                    },
                    x: {
                        ticks: { font: { size: 13, weight: 'bold' } },
                        grid: { display: false }
                    }
                },
                animation: {
                    duration: 800,
                    easing: 'easeOutQuart'
                }
            }
        });
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    new StockRankings();
});