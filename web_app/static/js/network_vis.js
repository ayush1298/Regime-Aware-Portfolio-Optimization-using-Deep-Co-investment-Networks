// static/js/network_vis.js
class NetworkVisualization {
    constructor() {
        this.svg = null;
        this.simulation = null;
        this.currentData = null;
        this.width = 0;
        this.height = 0;
        this.zoom = null;
        this.mainGroup = null;   // parent <g> that receives zoom transforms
        this.init();
        this.bindEvents();
    }

    init() {
        const container = d3.select('#networkViz');
        const containerNode = container.node();
        this.width = containerNode.clientWidth;
        this.height = containerNode.clientHeight;

        this.svg = container.append('svg')
            .attr('width', this.width)
            .attr('height', this.height);

        // Arrow-head marker definition (lives outside the zoom group)
        this.svg.append('defs').append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '-0 -5 10 10')
            .attr('refX', 13)
            .attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', 13)
            .attr('markerHeight', 13)
            .attr('xoverflow', 'visible')
            .append('svg:path')
            .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
            .attr('fill', '#999')
            .style('stroke', 'none');

        // ----- d3.zoom setup -----
        this.zoom = d3.zoom()
            .scaleExtent([0.15, 6])
            .on('zoom', (event) => {
                this.mainGroup.attr('transform', event.transform);
            });

        this.svg.call(this.zoom);

        // Primary <g> that holds ALL visual elements – zoomed uniformly
        this.mainGroup = this.svg.append('g').attr('class', 'main-group');

        this.loadNetworkData(2012);
    }

    bindEvents() {
        document.getElementById('yearSelect').addEventListener('change', (e) => {
            this.loadNetworkData(parseInt(e.target.value));
        });

        document.getElementById('searchBtn').addEventListener('click', () => {
            this.searchStock();
        });

        document.getElementById('stockSearch').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.searchStock();
            }
        });

        // ---- Zoom button listeners ----
        document.getElementById('zoomInBtn').addEventListener('click', () => {
            this.svg.transition().duration(350).call(this.zoom.scaleBy, 1.4);
        });

        document.getElementById('zoomOutBtn').addEventListener('click', () => {
            this.svg.transition().duration(350).call(this.zoom.scaleBy, 0.7);
        });

        document.getElementById('zoomResetBtn').addEventListener('click', () => {
            this.svg.transition().duration(500).call(
                this.zoom.transform, d3.zoomIdentity
            );
        });
    }

    async loadNetworkData(year) {
        document.getElementById('loading').style.display = 'block';
        
        try {
            const response = await fetch(`/api/network/${year}`);
            const data = await response.json();
            this.currentData = data;
            this.renderNetwork(data);
        } catch (error) {
            console.error('Error loading network data:', error);
        } finally {
            document.getElementById('loading').style.display = 'none';
        }
    }

    renderNetwork(data) {
        // Clear only the main group content (preserves defs & zoom binding)
        this.mainGroup.selectAll('*').remove();

        // Reset zoom transform
        this.svg.call(this.zoom.transform, d3.zoomIdentity);

        // ---- Populate the datalist for autocomplete ----
        const datalist = document.getElementById('stockList');
        datalist.innerHTML = '';
        data.nodes.forEach(n => {
            const opt = document.createElement('option');
            opt.value = n.id;
            opt.label = n.name || n.id;
            datalist.appendChild(opt);
        });

        // Scales
        const linkScale = d3.scaleLinear()
            .domain(d3.extent(data.links, d => d.weight))
            .range([1, 8]);

        const nodeScale = d3.scaleLinear()
            .domain(d3.extent(data.nodes, d => d.degree))
            .range([5, 20]);

        // Force simulation
        this.simulation = d3.forceSimulation(data.nodes)
            .force('link', d3.forceLink(data.links).id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2));

        // Links – inside mainGroup
        const link = this.mainGroup.append('g')
            .attr('class', 'links')
            .selectAll('line')
            .data(data.links)
            .enter().append('line')
            .attr('stroke-width', d => linkScale(d.weight))
            .attr('stroke', '#999')
            .attr('stroke-opacity', 0.6);

        // Nodes – inside mainGroup
        const node = this.mainGroup.append('g')
            .attr('class', 'nodes')
            .selectAll('circle')
            .data(data.nodes)
            .enter().append('circle')
            .attr('r', d => nodeScale(d.degree))
            .attr('fill', d => this.getNodeColor(d))
            .call(d3.drag()
                .on('start', (event, d) => this.dragstarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragended(event, d)));

        // Labels – inside mainGroup
        const labels = this.mainGroup.append('g')
            .attr('class', 'labels')
            .selectAll('text')
            .data(data.nodes.filter(d => d.degree > 10))
            .enter().append('text')
            .text(d => d.id)
            .attr('font-size', '10px')
            .attr('text-anchor', 'middle')
            .attr('dy', '.35em');

        // Click handler
        node.on('click', (event, d) => {
            this.showStockInfo(d);
            this.highlightNode(d);
        });

        // Tick
        this.simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);

            labels
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });
    }

    getNodeColor(node) {
        if (node.degree > 20) return '#e74c3c';
        if (node.degree > 10) return '#f39c12';
        return '#3498db';
    }

    showStockInfo(node) {
        const infoDiv = document.getElementById('stockInfo');
        const nameDiv = document.getElementById('stockName');
        const detailsDiv = document.getElementById('stockDetails');

        nameDiv.textContent = `${node.name} (${node.id})`;
        
        let detailsHTML = `
            <p><strong>Degree:</strong> ${node.degree}</p>
            <p><strong>Market Cap:</strong> $${(node.market_cap / 1000).toFixed(1)}B</p>
            <h6>Top Connections:</h6>
            <ul>
        `;

        node.top_connections.forEach(conn => {
            detailsHTML += `<li>${conn.ticker}: ${conn.weight.toFixed(3)}</li>`;
        });

        detailsHTML += '</ul>';
        detailsDiv.innerHTML = detailsHTML;
        infoDiv.style.display = 'block';
    }

    highlightNode(selectedNode) {
        this.mainGroup.selectAll('circle')
            .attr('stroke', d => d.id === selectedNode.id ? '#2c3e50' : 'none')
            .attr('stroke-width', d => d.id === selectedNode.id ? 3 : 0);
    }

    async searchStock() {
        const ticker = document.getElementById('stockSearch').value.toUpperCase();
        const year = document.getElementById('yearSelect').value;

        if (!ticker) return;

        try {
            const response = await fetch(`/api/search_stock?ticker=${ticker}&year=${year}`);
            const data = await response.json();
            
            if (data.error) {
                alert('Stock not found in the network');
                return;
            }

            this.showStockInfo(data);
            this.highlightNode(data);
            
            // Center on the found node using the zoom behaviour
            const nodeData = this.mainGroup.selectAll('circle').data().find(d => d.id === ticker);
            if (nodeData) {
                const transform = d3.zoomIdentity
                    .translate(this.width / 2 - nodeData.x * 1.5, this.height / 2 - nodeData.y * 1.5)
                    .scale(1.5);
                
                this.svg.transition().duration(750).call(
                    this.zoom.transform, transform
                );
            }
        } catch (error) {
            console.error('Error searching stock:', error);
        }
    }

    dragstarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    dragended(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    new NetworkVisualization();
});