import { Link } from "expo-router";
import { Text, View } from "react-native";

export default function Index() {
  return (
    <View className="flex-1 items-center justify-center bg-white">
      <Text className="text-xl font-bold text-blue-500">
        Crack Detection!!
      </Text>
      <Link href='/depthsensing'>LiDAR</Link>
    </View>
  );
}
