package availability

import (
	"math/rand"
	"strconv"
	"strings"
	"time"
)

const (
	driverType  = "DRIVER"
	vehicleType = "VEHICLE"
)

// GenerateResourceAvailability creates availability slots for all drivers (1..driverCount) and vehicles (1..vehicleCount).
// Each resource gets slotsPerResource slots with random valid_from/valid_to in the next daysAhead days.
func GenerateResourceAvailability(driverCount, vehicleCount, slotsPerResource, daysAhead int) []ResourceAvailability {
	out := make([]ResourceAvailability, 0, (driverCount+vehicleCount)*slotsPerResource)
	id := 1
	now := time.Now()

	for i := 1; i <= driverCount; i++ {
		slots := generateSlotsForResource(id, driverType, i, slotsPerResource, daysAhead, now)
		id += len(slots)
		out = append(out, slots...)
	}
	for i := 1; i <= vehicleCount; i++ {
		slots := generateSlotsForResource(id, vehicleType, i, slotsPerResource, daysAhead, now)
		id += len(slots)
		out = append(out, slots...)
	}
	return out
}

func generateSlotsForResource(nextID int, resourceType string, resourceID, count, daysAhead int, base time.Time) []ResourceAvailability {
	slots := make([]ResourceAvailability, 0, count)
	for i := 0; i < count; i++ {
		from := base.Add(time.Duration(rand.Intn(daysAhead*24)) * time.Hour)
		lengthHours := 2 + rand.Intn(10)
		to := from.Add(time.Duration(lengthHours) * time.Hour)
		status := Available
		if rand.Intn(3) == 0 {
			status = Reserved
		}
		notes := ""
		if status == Reserved && rand.Intn(2) == 0 {
			notes = "Reserved for order"
		}
		slots = append(slots, ResourceAvailability{
			ID:           nextID + i,
			ResourceType: resourceType,
			ResourceID:   resourceID,
			ValidFrom:    from.Format("2006-01-02 15:04:05"),
			ValidTo:      to.Format("2006-01-02 15:04:05"),
			Status:       status,
			Notes:        notes,
		})
	}
	return slots
}

// GenerateInsertStatements returns a single INSERT for resource_availability rows.
func GenerateInsertStatements(slots []ResourceAvailability) string {
	if len(slots) == 0 {
		return ""
	}
	var sb strings.Builder
	sb.Grow(len(slots) * 120)
	sb.WriteString("INSERT INTO resource_availability (id, resource_type, resource_id, valid_from, valid_to, status, notes) VALUES\n")
	for i, s := range slots {
		sb.WriteString("    (")
		sb.WriteString(strconv.Itoa(s.ID))
		sb.WriteString(", '")
		sb.WriteString(escapeSQL(s.ResourceType))
		sb.WriteString("', ")
		sb.WriteString(strconv.Itoa(s.ResourceID))
		sb.WriteString(", '")
		sb.WriteString(s.ValidFrom)
		sb.WriteString("', '")
		sb.WriteString(s.ValidTo)
		sb.WriteString("', '")
		sb.WriteString(string(s.Status))
		sb.WriteString("', ")
		writeNullableString(&sb, s.Notes)
		sb.WriteString(")")
		if i < len(slots)-1 {
			sb.WriteString(",\n")
		} else {
			sb.WriteString(";\n")
		}
	}
	return sb.String()
}

func escapeSQL(s string) string {
	return strings.ReplaceAll(s, "'", "''")
}

func writeNullableString(sb *strings.Builder, s string) {
	if s == "" {
		sb.WriteString("NULL")
		return
	}
	sb.WriteString("'")
	sb.WriteString(escapeSQL(s))
	sb.WriteString("'")
}
