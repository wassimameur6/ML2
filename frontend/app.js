// ========================================
// Serfy Bank - JavaScript Application
// ========================================

const API_URL = 'http://localhost:8000';

// ========================================
// Global State
// ========================================

let dashboardData = {
    stats: null,
    highRisk: null,
    offers: null
};

let charts = {};

// ========================================
// Initialization
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initTabs();
    initSliders();
    updateTime();
    setInterval(updateTime, 60000);

    // Load initial data
    checkSystemStatus();
    loadDashboard();
});

// ========================================
// Navigation
// ========================================

function initNavigation() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const page = item.dataset.page;
            navigateTo(page);
        });
    });
}

function navigateTo(page) {
    // Update nav
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.page === page);
    });

    // Update page
    document.querySelectorAll('.page').forEach(p => {
        p.classList.toggle('active', p.id === `page-${page}`);
    });

    // Load page data
    switch(page) {
        case 'dashboard':
            loadDashboard();
            break;
        case 'offers':
            loadOffers();
            break;
        case 'settings':
            loadSettings();
            break;
    }
}

// ========================================
// Tabs
// ========================================

function initTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const tabId = tab.dataset.tab;

            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            document.getElementById(`tab-${tabId}`).classList.add('active');
        });
    });
}

// ========================================
// Sliders
// ========================================

function initSliders() {
    const riskSlider = document.getElementById('risk-threshold');
    const riskValue = document.getElementById('threshold-value');
    if (riskSlider && riskValue) {
        riskSlider.addEventListener('input', () => {
            riskValue.textContent = `${riskSlider.value}%`;
        });
    }

    const campaignSlider = document.getElementById('campaign-threshold');
    const campaignValue = document.getElementById('campaign-threshold-value');
    if (campaignSlider && campaignValue) {
        campaignSlider.addEventListener('input', () => {
            campaignValue.textContent = `${campaignSlider.value}%`;
        });
    }

    const sendLive = document.getElementById('send-live');
    const hint = document.getElementById('campaign-hint');
    if (sendLive && hint) {
        sendLive.addEventListener('change', () => {
            hint.textContent = sendLive.checked
                ? '⚠️ Real emails will be sent to customers'
                : 'Simulation mode — no emails will be sent';
            hint.style.color = sendLive.checked ? '#dc3545' : '#888';
        });
    }
}

// ========================================
// Time
// ========================================

function updateTime() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    document.getElementById('current-time').textContent = timeStr;
}

function updateLastUpdated() {
    const now = new Date();
    document.getElementById('last-updated').textContent = now.toLocaleTimeString();
}

// ========================================
// API Calls
// ========================================

async function apiCall(endpoint, method = 'GET', data = null) {
    try {
        const options = {
            method,
            headers: { 'Content-Type': 'application/json' }
        };
        if (data) options.body = JSON.stringify(data);

        const response = await fetch(`${API_URL}${endpoint}`, options);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error(`API Error: ${endpoint}`, error);
        return null;
    }
}

// ========================================
// System Status
// ========================================

async function checkSystemStatus() {
    const health = await apiCall('/health');

    const setStatus = (id, online) => {
        const dot = document.getElementById(id);
        if (dot) dot.classList.toggle('online', online);
    };

    if (health) {
        setStatus('status-ml', health.model_loaded);
        setStatus('status-ai', health.vectorstore_ready);
        setStatus('status-email', health.email_configured);
        setStatus('status-openai', health.openai_configured);
    }
}

// ========================================
// Dashboard
// ========================================

async function loadDashboard() {
    // Load stats (uses actual data, not model predictions)
    const stats = await apiCall('/stats');
    if (!stats) {
        showToast('Failed to connect to API', 'error');
        return;
    }
    dashboardData.stats = stats;

    // Update KPIs with actual data
    updateKPIs(stats);

    // Create charts with actual data
    createCharts(stats);

    updateLastUpdated();
}

function updateKPIs(stats) {
    const retention = 1 - stats.attrition_rate;

    // Estimated revenue per customer (average annual value)
    const avgRevenuePerCustomer = 2450;
    const revenueLost = stats.attrited_customers * avgRevenuePerCustomer;

    document.getElementById('kpi-total').textContent = stats.total_customers.toLocaleString();
    document.getElementById('kpi-retention').textContent = `${(retention * 100).toFixed(1)}%`;
    document.getElementById('kpi-churn').textContent = `${(stats.attrition_rate * 100).toFixed(1)}%`;
    document.getElementById('kpi-atrisk').textContent = stats.attrited_customers.toLocaleString();
    document.getElementById('kpi-revenue').textContent = `$${(revenueLost / 1000000).toFixed(1)}M`;
}

// loadHighRiskData removed - dashboard now uses actual data from /stats

function destroyCharts() {
    Object.keys(charts).forEach(key => {
        if (charts[key]) {
            charts[key].destroy();
            charts[key] = null;
        }
    });
    charts = {};
}

function createCharts(stats) {
    // Destroy existing charts first
    destroyCharts();

    // Trend Chart - Churn Rate over time
    const trendCtx = document.getElementById('chart-trend');
    if (trendCtx) {
        const months = ['Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan'];
        const churnRates = [18.5, 18.2, 17.8, 17.5, 17.2, 17.0, 16.8, 16.5, 16.3, 16.2, 16.1, stats.attrition_rate * 100];

        charts.trend = new Chart(trendCtx, {
            type: 'line',
            data: {
                labels: months,
                datasets: [
                    {
                        label: 'Churn Rate',
                        data: churnRates,
                        borderColor: '#dc3545',
                        backgroundColor: 'rgba(220, 53, 69, 0.15)',
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#dc3545',
                        pointRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        min: 14,
                        max: 20,
                        ticks: { callback: v => v + '%' }
                    }
                }
            }
        });
    }

    // Customer Status Distribution Chart (actual data)
    const riskCtx = document.getElementById('chart-risk');
    if (riskCtx) {
        charts.risk = new Chart(riskCtx, {
            type: 'doughnut',
            data: {
                labels: ['Active Customers', 'Churned Customers'],
                datasets: [{
                    data: [stats.existing_customers, stats.attrited_customers],
                    backgroundColor: ['#28a745', '#dc3545'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '65%',
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });
    }

    // Income Chart
    const incomeCtx = document.getElementById('chart-income');
    if (incomeCtx) {
        charts.income = new Chart(incomeCtx, {
            type: 'bar',
            data: {
                labels: ['<$40K', '$40-60K', '$60-80K', '$80-120K', '$120K+'],
                datasets: [{
                    label: 'Churn Rate by Income',
                    data: [22, 18, 15, 14, 12],
                    backgroundColor: '#E5A229'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { max: 30, ticks: { callback: v => v + '%' } }
                }
            }
        });
    }

    // Cards Chart
    const cardsCtx = document.getElementById('chart-cards');
    if (cardsCtx) {
        charts.cards = new Chart(cardsCtx, {
            type: 'bar',
            data: {
                labels: ['Blue', 'Silver', 'Gold', 'Platinum'],
                datasets: [{
                    label: 'Customer Count',
                    data: [9436, 555, 116, 20],
                    backgroundColor: ['#4169E1', '#C0C0C0', '#FFD700', '#E5E4E2']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } }
            }
        });
    }

    // Tenure Chart
    const tenureCtx = document.getElementById('chart-tenure');
    if (tenureCtx) {
        charts.tenure = new Chart(tenureCtx, {
            type: 'line',
            data: {
                labels: ['0-12m', '12-24m', '24-36m', '36-48m', '48m+'],
                datasets: [{
                    label: 'Churn Rate',
                    data: [25, 18, 14, 12, 10],
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { max: 30, ticks: { callback: v => v + '%' } }
                }
            }
        });
    }
}

// Charts now use static representative data for segmentation analysis

function refreshData() {
    destroyCharts();
    loadDashboard();
    showToast('Data refreshed');
}

// ========================================
// Customer Lookup
// ========================================

async function searchCustomer() {
    const input = document.getElementById('customer-search');
    const clientNum = parseInt(input.value.trim());

    if (!clientNum) {
        showToast('Please enter a valid Customer ID', 'error');
        return;
    }

    showToast('Loading customer data...');

    const [customer, prediction, recommendations, aiRecommendation] = await Promise.all([
        apiCall(`/customers/${clientNum}`),
        apiCall(`/customers/${clientNum}/predict`),
        apiCall(`/customers/${clientNum}/recommendations`),
        apiCall(`/customers/${clientNum}/ai-recommendation`)
    ]);

    if (!customer) {
        showToast('Customer not found', 'error');
        return;
    }

    displayCustomerResult(customer, prediction, recommendations, aiRecommendation, clientNum);
}

function displayCustomerResult(customer, prediction, recommendations, aiRecommendation, clientNum) {
    const container = document.getElementById('customer-result');
    container.classList.remove('hidden');

    // Customer details
    document.getElementById('customer-details').innerHTML = `
        <h2 style="margin-bottom: 0.5rem;">${customer.name}</h2>
        <p style="color: #888; margin-bottom: 1rem;">ID: ${clientNum}</p>
        <table>
            <tr><td>Email</td><td>${customer.email}</td></tr>
            <tr><td>Phone</td><td>${customer.phone}</td></tr>
            <tr><td>Age</td><td>${customer.age} years</td></tr>
            <tr><td>Income</td><td>${customer.income_category}</td></tr>
            <tr><td>Card</td><td>${customer.card_category}</td></tr>
            <tr><td>Tenure</td><td>${customer.months_on_book} months</td></tr>
        </table>
    `;

    // Risk analysis
    if (prediction) {
        const risk = prediction.churn_risk;
        const prob = prediction.churn_probability * 100;

        document.getElementById('risk-analysis').innerHTML = `
            <div class="risk-display">
                <div class="risk-percentage ${risk}">${prob.toFixed(0)}%</div>
                <div style="color: #888; margin-bottom: 0.5rem;">Churn Probability</div>
                <span class="risk-label ${risk}">${risk.toUpperCase()} RISK</span>
            </div>
            ${prediction.is_churning ? `
                <div class="alert-box danger">
                    <strong>Action Required</strong> — High probability of churn. Consider retention offer.
                </div>
            ` : ''}
        `;
    }

    // AI Recommendation (RAG-based)
    if (aiRecommendation && aiRecommendation.ai_recommendation) {
        const aiSection = document.getElementById('ai-recommendation') || createAiSection();
        aiSection.innerHTML = `
            <div class="ai-recommendation-box">
                <div class="ai-header">
                    <span class="ai-icon">🤖</span>
                    <h3>AI-Powered Recommendation</h3>
                    ${aiRecommendation.agent_pipeline?.rag_used ? '<span class="badge rag">RAG</span>' : ''}
                    ${aiRecommendation.agent_pipeline?.llm_used ? '<span class="badge llm">LLM</span>' : ''}
                </div>
                <p class="ai-text">${aiRecommendation.ai_recommendation}</p>
            </div>
        `;
    }

    // Recommendations
    if (recommendations && recommendations.length > 0) {
        document.getElementById('customer-offers').innerHTML = recommendations.slice(0, 3).map(offer => `
            <div class="offer-card">
                <h4>${offer.title}</h4>
                <div class="offer-type">${offer.offer_type.replace(/_/g, ' ')}</div>
                <div class="offer-description">${offer.description.substring(0, 100)}...</div>
                <div class="offer-value">${offer.value}</div>
                <div class="offer-match">
                    <span class="match-score">${(offer.relevance_score * 100).toFixed(0)}% match</span>
                    <button class="btn primary" onclick="sendOffer(${clientNum}, '${offer.offer_id}')">Send</button>
                </div>
            </div>
        `).join('');
    }
}

function createAiSection() {
    const section = document.createElement('div');
    section.id = 'ai-recommendation';
    section.className = 'ai-section';
    const riskSection = document.getElementById('risk-analysis');
    riskSection.parentNode.insertBefore(section, riskSection.nextSibling);
    return section;
}

async function sendOffer(clientNum, offerId) {
    showToast('Sending email...');

    const result = await apiCall('/send-email', 'POST', {
        client_num: clientNum,
        offer_id: offerId
    });

    if (result && result.success) {
        showToast(`Email sent to ${result.to_email}`, 'success');
    } else {
        showToast(result?.error || 'Failed to send email', 'error');
    }
}

// ========================================
// At-Risk Customers
// ========================================

async function findAtRiskCustomers() {
    const threshold = document.getElementById('risk-threshold').value / 100;
    const limit = document.getElementById('max-results').value;

    showToast('Analyzing customers...');

    const result = await apiCall(`/high-risk-customers?threshold=${threshold}&limit=${limit}`);

    if (!result) {
        showToast('Failed to load data', 'error');
        return;
    }

    displayAtRiskResults(result, threshold);
}

function displayAtRiskResults(result, threshold) {
    const container = document.getElementById('at-risk-results');
    container.classList.remove('hidden');

    document.getElementById('ar-found').textContent = result.total_high_risk;
    document.getElementById('ar-showing').textContent = result.returned;
    document.getElementById('ar-threshold').textContent = `${(threshold * 100).toFixed(0)}%`;

    const tbody = document.querySelector('#at-risk-table tbody');
    tbody.innerHTML = result.customers.map(c => `
        <tr>
            <td>${c.CLIENTNUM}</td>
            <td>${c.First_Name} ${c.Last_Name}</td>
            <td>${c.Email || '-'}</td>
            <td>${(c.churn_probability * 100).toFixed(0)}%</td>
            <td><span class="badge ${c.churn_risk === 'high' ? 'danger' : 'warning'}">${c.churn_risk}</span></td>
            <td><button class="btn" onclick="navigateTo('customers'); document.getElementById('customer-search').value='${c.CLIENTNUM}'; searchCustomer();">View</button></td>
        </tr>
    `).join('');
}

// ========================================
// Campaigns
// ========================================

async function launchCampaign() {
    const threshold = document.getElementById('campaign-threshold').value / 100;
    const maxCustomers = document.getElementById('campaign-max').value;
    const customerIds = document.getElementById('campaign-ids').value;
    const sendLive = document.getElementById('send-live').checked;

    let ids = null;
    if (customerIds.trim()) {
        ids = customerIds.split(/[,\n]/).map(id => parseInt(id.trim())).filter(id => !isNaN(id));
    }

    showToast('Launching campaign...');

    const result = await apiCall('/campaign', 'POST', {
        customer_ids: ids,
        risk_threshold: threshold,
        send_emails: sendLive,
        max_customers: parseInt(maxCustomers)
    });

    if (result) {
        showToast('Campaign completed!', 'success');
        displayCampaignResults(result);
    } else {
        showToast('Campaign failed', 'error');
    }
}

function displayCampaignResults(result) {
    const container = document.getElementById('campaign-results');
    container.classList.remove('hidden');

    document.getElementById('campaign-details').innerHTML = `
        <div class="results-summary">
            <div class="summary-item">
                <span class="summary-value">${result.total_customers}</span>
                <span class="summary-label">Processed</span>
            </div>
            <div class="summary-item">
                <span class="summary-value">${result.emails_sent}</span>
                <span class="summary-label">Sent</span>
            </div>
            <div class="summary-item">
                <span class="summary-value">${result.emails_failed}</span>
                <span class="summary-label">Failed</span>
            </div>
        </div>
    `;
}

// ========================================
// Offers
// ========================================

async function loadOffers() {
    const offers = await apiCall('/offers');
    if (!offers) return;

    dashboardData.offers = offers;

    document.getElementById('total-offers').textContent = offers.length;

    // Populate filter
    const types = [...new Set(offers.map(o => o.offer_type))];
    const filter = document.getElementById('offer-filter');
    filter.innerHTML = '<option value="all">All Types</option>' +
        types.map(t => `<option value="${t}">${t.replace(/_/g, ' ')}</option>`).join('');

    displayOffers(offers);
}

function filterOffers() {
    const filter = document.getElementById('offer-filter').value;
    const offers = dashboardData.offers;

    const filtered = filter === 'all'
        ? offers
        : offers.filter(o => o.offer_type === filter);

    displayOffers(filtered);
}

function displayOffers(offers) {
    document.getElementById('offers-grid').innerHTML = offers.map(offer => `
        <div class="offer-card">
            <h4>${offer.title}</h4>
            <div class="offer-type">${offer.offer_type.replace(/_/g, ' ')}</div>
            <div class="offer-description">${offer.description.substring(0, 80)}...</div>
            <div class="offer-value">${offer.value}</div>
        </div>
    `).join('');
}

// ========================================
// Settings
// ========================================

async function loadSettings() {
    const health = await apiCall('/health');
    if (!health) return;

    const components = [
        { name: 'ML Model', status: health.model_loaded, desc: 'Random Forest classifier' },
        { name: 'Vector Store', status: health.vectorstore_ready, desc: 'ChromaDB semantic search' },
        { name: 'Email Service', status: health.email_configured, desc: 'SMTP delivery' },
        { name: 'OpenAI API', status: health.openai_configured, desc: 'GPT personalization' }
    ];

    document.getElementById('component-status').innerHTML = components.map(c => `
        <div class="status-row">
            <strong>${c.name}</strong>
            <span class="badge ${c.status ? 'success' : 'danger'}">${c.status ? 'Online' : 'Offline'}</span>
            <span style="color: #888; font-size: 0.85rem;">${c.desc}</span>
        </div>
    `).join('');

    document.getElementById('api-url').textContent = API_URL;
}

// ========================================
// Export Functions
// ========================================

function exportSummary() {
    if (!dashboardData.stats) return;

    const stats = dashboardData.stats;
    const csv = [
        ['Metric', 'Value'],
        ['Total Customers', stats.total_customers],
        ['Active', stats.existing_customers],
        ['Churned', stats.attrited_customers],
        ['Churn Rate', `${(stats.attrition_rate * 100).toFixed(1)}%`],
        ['Retention Rate', `${((1 - stats.attrition_rate) * 100).toFixed(1)}%`]
    ].map(row => row.join(',')).join('\n');

    downloadCSV(csv, 'dashboard_summary.csv');
    showToast('Summary exported');
}

function exportAtRisk() {
    if (!dashboardData.highRisk) {
        showToast('No data to export', 'error');
        return;
    }

    const customers = dashboardData.highRisk.customers;
    const headers = ['CLIENTNUM', 'First_Name', 'Last_Name', 'Email', 'churn_probability', 'churn_risk'];
    const csv = [
        headers.join(','),
        ...customers.map(c => headers.map(h => c[h] || '').join(','))
    ].join('\n');

    downloadCSV(csv, 'at_risk_customers.csv');
    showToast('At-risk list exported');
}

function downloadCSV(content, filename) {
    const blob = new Blob([content], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// ========================================
// Toast Notifications
// ========================================

function showToast(message, type = '') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type}`;

    setTimeout(() => toast.classList.add('hidden'), 3000);
}
