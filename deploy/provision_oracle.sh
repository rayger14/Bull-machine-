#!/usr/bin/env bash
# =============================================================================
# provision_oracle.sh — Auto-create Oracle Cloud VM for Bull Machine
#
# Tries multiple regions/ADs to find available ARM Ampere capacity.
# Requires: OCI CLI configured (~/.oci/config) with API key uploaded.
#
# Usage:
#   bash deploy/provision_oracle.sh
# =============================================================================
set -euo pipefail

OCI="/Users/raymondghandchi/Library/Python/3.9/bin/oci"
SSH_PUB_KEY=$(cat ~/.ssh/oracle_bullmachine.pub)

# Regions to try (Always Free eligible, ordered by proximity to US West)
REGIONS=(
    "us-sanjose-1"
    "us-phoenix-1"
    "us-ashburn-1"
    "ca-toronto-1"
)

echo "=== Bull Machine — Oracle Cloud VM Provisioner ==="
echo ""

# Verify CLI works
echo "--- Verifying OCI CLI authentication ---"
if ! $OCI iam region list --output table 2>/dev/null | head -5; then
    echo "ERROR: OCI CLI not authenticated. Upload your API key to Oracle Console first."
    echo "  File to upload: ~/Desktop/oci_api_key_public.pem"
    echo "  Location: Profile > My Profile > API keys > Add API key > Choose public key file"
    exit 1
fi
echo "  OCI CLI authenticated."
echo ""

# Get home region's tenancy info
TENANCY_ID=$($OCI iam tenancy get --tenancy-id "$($OCI iam region-subscription list --query 'data[0]."tenancy-id"' --raw-output 2>/dev/null)" --query 'data."home-region-key"' --raw-output 2>/dev/null || echo "")

create_vm() {
    local REGION=$1
    echo ""
    echo "=== Trying region: ${REGION} ==="

    # Update CLI to use this region
    export OCI_CLI_REGION="${REGION}"

    # Get compartment (root = tenancy)
    COMPARTMENT=$($OCI iam compartment list \
        --compartment-id-in-subtree true \
        --query 'data[0].id' --raw-output 2>/dev/null || echo "")

    if [ -z "$COMPARTMENT" ]; then
        # Use tenancy as compartment (root)
        COMPARTMENT=$(grep "tenancy=" ~/.oci/config | head -1 | cut -d= -f2)
    fi
    echo "  Compartment: ${COMPARTMENT}"

    # Get availability domains
    ADS=$($OCI iam availability-domain list \
        --compartment-id "${COMPARTMENT}" \
        --query 'data[*].name' --raw-output 2>/dev/null || echo "[]")

    if [ "$ADS" = "[]" ] || [ -z "$ADS" ]; then
        echo "  No availability domains found. Skipping."
        return 1
    fi
    echo "  Availability domains: ${ADS}"

    # Get Ubuntu 22.04 aarch64 image
    IMAGE_ID=$($OCI compute image list \
        --compartment-id "${COMPARTMENT}" \
        --operating-system "Canonical Ubuntu" \
        --operating-system-version "22.04" \
        --shape "VM.Standard.A1.Flex" \
        --sort-by TIMECREATED --sort-order DESC \
        --query 'data[0].id' --raw-output 2>/dev/null || echo "")

    if [ -z "$IMAGE_ID" ] || [ "$IMAGE_ID" = "null" ]; then
        # Try minimal image
        IMAGE_ID=$($OCI compute image list \
            --compartment-id "${COMPARTMENT}" \
            --operating-system "Canonical Ubuntu" \
            --operating-system-version "22.04 Minimal aarch64" \
            --sort-by TIMECREATED --sort-order DESC \
            --query 'data[0].id' --raw-output 2>/dev/null || echo "")
    fi

    if [ -z "$IMAGE_ID" ] || [ "$IMAGE_ID" = "null" ]; then
        echo "  No Ubuntu 22.04 ARM image found. Skipping."
        return 1
    fi
    echo "  Image: ${IMAGE_ID}"

    # Create VCN
    echo "  Creating VCN..."
    VCN_ID=$($OCI network vcn create \
        --compartment-id "${COMPARTMENT}" \
        --display-name "bull-machine-vcn" \
        --cidr-blocks '["10.0.0.0/16"]' \
        --query 'data.id' --raw-output 2>/dev/null || echo "")

    if [ -z "$VCN_ID" ] || [ "$VCN_ID" = "null" ]; then
        echo "  Failed to create VCN. Skipping."
        return 1
    fi
    echo "  VCN: ${VCN_ID}"

    # Create Internet Gateway
    echo "  Creating Internet Gateway..."
    IGW_ID=$($OCI network internet-gateway create \
        --compartment-id "${COMPARTMENT}" \
        --vcn-id "${VCN_ID}" \
        --display-name "bull-machine-igw" \
        --is-enabled true \
        --query 'data.id' --raw-output 2>/dev/null || echo "")

    # Get default route table
    RT_ID=$($OCI network route-table list \
        --compartment-id "${COMPARTMENT}" \
        --vcn-id "${VCN_ID}" \
        --query 'data[0].id' --raw-output 2>/dev/null || echo "")

    # Add default route to internet gateway
    if [ -n "$RT_ID" ] && [ "$RT_ID" != "null" ] && [ -n "$IGW_ID" ] && [ "$IGW_ID" != "null" ]; then
        $OCI network route-table update \
            --rt-id "${RT_ID}" \
            --route-rules "[{\"destination\":\"0.0.0.0/0\",\"destinationType\":\"CIDR_BLOCK\",\"networkEntityId\":\"${IGW_ID}\"}]" \
            --force 2>/dev/null || true
    fi

    # Get default security list and add port 8080
    SL_ID=$($OCI network security-list list \
        --compartment-id "${COMPARTMENT}" \
        --vcn-id "${VCN_ID}" \
        --query 'data[0].id' --raw-output 2>/dev/null || echo "")

    if [ -n "$SL_ID" ] && [ "$SL_ID" != "null" ]; then
        # Add SSH (22) and FreqUI (8080) ingress rules
        INGRESS='[{"source":"0.0.0.0/0","protocol":"6","tcpOptions":{"destinationPortRange":{"min":22,"max":22}}},{"source":"0.0.0.0/0","protocol":"6","tcpOptions":{"destinationPortRange":{"min":8080,"max":8080}}}]'
        EGRESS='[{"destination":"0.0.0.0/0","protocol":"all"}]'
        $OCI network security-list update \
            --security-list-id "${SL_ID}" \
            --ingress-security-rules "${INGRESS}" \
            --egress-security-rules "${EGRESS}" \
            --force 2>/dev/null || true
    fi

    # Try each AD
    for AD_NAME in $($OCI iam availability-domain list \
        --compartment-id "${COMPARTMENT}" \
        --query 'data[*].name' --raw-output 2>/dev/null | tr -d '[]",' | tr ' ' '\n' | grep -v '^$'); do

        echo "  Trying AD: ${AD_NAME}..."

        # Create subnet in this AD
        SUBNET_ID=$($OCI network subnet create \
            --compartment-id "${COMPARTMENT}" \
            --vcn-id "${VCN_ID}" \
            --availability-domain "${AD_NAME}" \
            --display-name "bull-machine-subnet-${AD_NAME##*:}" \
            --cidr-block "10.0.${RANDOM:0:1}.0/24" \
            --query 'data.id' --raw-output 2>/dev/null || echo "")

        if [ -z "$SUBNET_ID" ] || [ "$SUBNET_ID" = "null" ]; then
            echo "    Failed to create subnet. Trying next AD..."
            continue
        fi

        # Wait for subnet to be available
        sleep 5

        # Launch instance
        echo "  Launching VM.Standard.A1.Flex (2 OCPU, 12 GB)..."
        INSTANCE_JSON=$($OCI compute instance launch \
            --compartment-id "${COMPARTMENT}" \
            --availability-domain "${AD_NAME}" \
            --shape "VM.Standard.A1.Flex" \
            --shape-config '{"ocpus":2,"memoryInGBs":12}' \
            --image-id "${IMAGE_ID}" \
            --subnet-id "${SUBNET_ID}" \
            --display-name "bull-machine-bot" \
            --assign-public-ip true \
            --ssh-authorized-keys-file <(echo "${SSH_PUB_KEY}") \
            --boot-volume-size-in-gbs 50 \
            2>&1 || echo "FAILED")

        if echo "$INSTANCE_JSON" | grep -q "FAILED\|OutOfCapacity\|out of capacity\|InternalError\|LimitExceeded"; then
            echo "    Out of capacity in ${AD_NAME}. Trying next..."
            # Clean up subnet
            $OCI network subnet delete --subnet-id "${SUBNET_ID}" --force 2>/dev/null || true
            continue
        fi

        INSTANCE_ID=$(echo "$INSTANCE_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['id'])" 2>/dev/null || echo "")

        if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "null" ]; then
            echo "    Launch failed. Trying next AD..."
            continue
        fi

        echo ""
        echo "  SUCCESS! Instance launching: ${INSTANCE_ID}"
        echo "  Waiting for public IP..."

        # Wait for running state
        sleep 30

        # Get VNIC attachment
        VNIC_ID=$($OCI compute vnic-attachment list \
            --compartment-id "${COMPARTMENT}" \
            --instance-id "${INSTANCE_ID}" \
            --query 'data[0]."vnic-id"' --raw-output 2>/dev/null || echo "")

        PUBLIC_IP=""
        for i in $(seq 1 12); do
            if [ -n "$VNIC_ID" ] && [ "$VNIC_ID" != "null" ]; then
                PUBLIC_IP=$($OCI network vnic get \
                    --vnic-id "${VNIC_ID}" \
                    --query 'data."public-ip"' --raw-output 2>/dev/null || echo "")
            fi
            if [ -n "$PUBLIC_IP" ] && [ "$PUBLIC_IP" != "null" ] && [ "$PUBLIC_IP" != "None" ]; then
                break
            fi
            echo "    Waiting for IP... (${i}/12)"
            sleep 10
        done

        if [ -n "$PUBLIC_IP" ] && [ "$PUBLIC_IP" != "null" ] && [ "$PUBLIC_IP" != "None" ]; then
            echo ""
            echo "==========================================="
            echo " VM CREATED SUCCESSFULLY!"
            echo "==========================================="
            echo " Region:    ${REGION}"
            echo " AD:        ${AD_NAME}"
            echo " Public IP: ${PUBLIC_IP}"
            echo " Instance:  ${INSTANCE_ID}"
            echo ""
            echo " Next steps:"
            echo "   1. Wait 2 minutes for VM to fully boot"
            echo "   2. SSH in:  ssh -i ~/.ssh/oracle_bullmachine ubuntu@${PUBLIC_IP}"
            echo "   3. Run setup: cd /home/ubuntu && git clone https://github.com/rayger14/Bull-machine-.git && cd Bull-machine- && bash deploy/setup_oracle.sh"
            echo ""

            # Update deploy.sh with the real IP
            DEPLOY_SCRIPT="/Users/raymondghandchi/Bull-machine-/Bull-machine-/deploy/deploy.sh"
            if [ -f "$DEPLOY_SCRIPT" ]; then
                sed -i '' "s/YOUR_SERVER_IP/${PUBLIC_IP}/" "$DEPLOY_SCRIPT"
                echo " deploy.sh updated with IP: ${PUBLIC_IP}"
            fi

            # Save connection info
            cat > "/Users/raymondghandchi/Bull-machine-/Bull-machine-/deploy/.server_info" << SERVERINFO
SERVER_IP=${PUBLIC_IP}
INSTANCE_ID=${INSTANCE_ID}
REGION=${REGION}
AD=${AD_NAME}
VCN_ID=${VCN_ID}
SUBNET_ID=${SUBNET_ID}
CREATED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
SERVERINFO
            echo " Server info saved to deploy/.server_info"

            return 0
        else
            echo "    Could not get public IP. Check Oracle Console."
            return 0
        fi
    done

    # Clean up VCN if no AD worked
    echo "  No capacity in any AD for ${REGION}. Cleaning up VCN..."
    # Delete subnets first
    for SN in $($OCI network subnet list --compartment-id "${COMPARTMENT}" --vcn-id "${VCN_ID}" --query 'data[*].id' --raw-output 2>/dev/null | tr -d '[]",' | tr ' ' '\n' | grep -v '^$'); do
        $OCI network subnet delete --subnet-id "$SN" --force 2>/dev/null || true
    done
    sleep 10
    # Delete IGW
    if [ -n "$IGW_ID" ] && [ "$IGW_ID" != "null" ]; then
        # Clear route table first
        $OCI network route-table update --rt-id "${RT_ID}" --route-rules '[]' --force 2>/dev/null || true
        sleep 5
        $OCI network internet-gateway delete --ig-id "${IGW_ID}" --force 2>/dev/null || true
    fi
    sleep 5
    $OCI network vcn delete --vcn-id "${VCN_ID}" --force 2>/dev/null || true

    return 1
}

# Try each region
SUCCESS=false
for REGION in "${REGIONS[@]}"; do
    if create_vm "$REGION"; then
        SUCCESS=true
        break
    fi
done

if [ "$SUCCESS" = false ]; then
    echo ""
    echo "==========================================="
    echo " NO CAPACITY FOUND"
    echo "==========================================="
    echo ""
    echo " ARM Ampere instances are out of capacity in all tried regions."
    echo " Options:"
    echo "   1. Wait and retry later (capacity fluctuates)"
    echo "   2. Try other regions by editing REGIONS in this script"
    echo "   3. Use the Oracle Console to try manually"
    echo ""
    echo " To retry: bash deploy/provision_oracle.sh"
fi
